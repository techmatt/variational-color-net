
local imageLoader = require('imageLoader')
local torchUtil = require('torchUtil')

local debugBatchIndices = {[500]=true, [6000]=true, [20000]=true}
-- local debugBatchIndices = {[5]=true}
--local debugBatchIndices = {}

-- Setup a reused optimization state (for adam/sgd).
local optimState = {
    learningRate = 0.0
}

local function paramsForEpoch(epoch)
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     1,   1e-3,   0 },
        {  2,     2,   1e-3,   0 },
        {  3,     3,   5e-4,   0 },
        {  4,     10,   4e-5,   0 },
        { 11,     20,   2e-5,   0 },
        { 21,     30,   1e-5,   0 },
        { 31,     40,   5e-6,   0 },
        { 41,    1e8,   1e-6,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- Stuff for logging
local trainLogger = nil
local batchNumber               -- Current batch in current epoch
local totalBatchCount = 0       -- Total # of batches across all epochs
local epochStats = {}

-- GPU inputs (preallocate)
local grayscaleInputs = torch.CudaTensor()
local colorTargets = torch.CudaTensor()

-- Parameters for training (will be filled in during first training iteration)
local parameters, gradParameters = nil, nil

-- 4. trainSuperBatch - Used by train() to train a superbatch.
local function trainSuperBatch(model, imgLoader, opt, epoch)
    if parameters == nil then
        parameters, gradParameters = model.trainNet:getParameters()
    end
    
    cutorch.synchronize()

    local dataLoadingTime = 0
    local startTime = os.clock()

    local reconLossSum, kldLossSum, totalLossSum = 0, 0, 0
    local feval = function(x)
        model.trainNet:zeroGradParameters()
        
        for superBatch = 1, opt.superBatches do
            local loadTimeStart = os.clock()
            local batch = imageLoader.sampleBatch(imgLoader)
            local loadTimeEnd = os.clock()
            dataLoadingTime = dataLoadingTime + (loadTimeEnd - loadTimeStart)

            -- Split images into L and a,b
            local grayscale = batch.grayscaleInputs     -- Using full-size Y instead of L b/c it's cheaper
            local thumbs = batch.normalizedThumbnails
            local color = thumbs[{ {},{2,3},{},{} }]
            
            -- transfer over to GPU
            grayscaleInputs:resize(grayscale:size()):copy(grayscale)
            colorTargets:resize(color:size()):copy(color)
            
            local outputLoss = model.trainNet:forward({grayscaleInputs, colorTargets})
            -- Trying to figure out how much to weight these things...
            local reconLossWeight = 10
            -- local reconLossWeight = 1
            local kldLossWeight = 1
            outputLoss[1]:mul(reconLossWeight)
            outputLoss[2]:mul(kldLossWeight)

            local reconLoss = outputLoss[1][1]
            local kldLoss = outputLoss[2][1]
            
            reconLossSum = reconLossSum + reconLoss
            kldLossSum = kldLossSum + kldLoss
            totalLossSum = totalLossSum + reconLoss + kldLoss
            
            model.trainNet:backward({grayscaleInputs, colorTargets}, outputLoss)
            
            if superBatch == 1 and debugBatchIndices[totalBatchCount] then
                torchUtil.dumpGraph(model.trainNet, opt.outDir .. 'graphDump' .. totalBatchCount .. '.csv')
            end

            -- Output test samples
            if superBatch == 1 and totalBatchCount % 100 == 0 then
            
                model.testNet:evaluate()
            
                -- Save ground truth RGB image
                local groundTruth = batch.images[1]
                image.save(opt.outDir .. 'samples/iter' .. totalBatchCount .. '_groundTruth.jpg', groundTruth)
                
                collectgarbage()

                -- Copy image #1 into the entire batch for grayscaleInputs
                -- This allows us to output N random samples from the network, where N = batch size
                for batchIndex = 2, opt.batchSize do
                    grayscaleInputs[batchIndex]:copy(grayscaleInputs[1])
                end
                
                local predictions = model.testNet:forward(grayscaleInputs)

                -- Compute ground truth L
                local groundTruthLab = torchUtil.normalizeLab(image.rgb2lab(groundTruth:clone()))
                local norm = groundTruthLab:norm()
                for sampleIndex = 1, opt.numTestSamples do
                    local prediction = predictions[sampleIndex]
                    local predictionCPU = torch.Tensor(prediction:size()):copy(prediction)
                    -- Rescale color to be same size as groundTruthL, then combine
                    predictionCPU = image.scale(predictionCPU, groundTruthLab:size()[2], groundTruthLab:size()[3])
                    local fullColorPred = torch.Tensor(groundTruthLab:size())
                    fullColorPred[1]:copy(groundTruthLab[1])
                    fullColorPred[2] = predictionCPU[1]
                    fullColorPred[3] = predictionCPU[2]
                    fullColorPred = image.lab2rgb(torchUtil.denormalizeLab(fullColorPred))
                    image.save(opt.outDir .. 'samples/iter' .. totalBatchCount .. '_sample' .. sampleIndex .. '_predictedGuideRGBSmall.jpg', fullColorPred)
                end
                
                model.testNet:training()
            end
        end
        
        return totalLossSum, gradParameters
    end
    optim.adam(feval, parameters, optimState)

    cutorch.synchronize()
    batchNumber = batchNumber + 1
    
    epochStats.total = epochStats.total + totalLossSum
    epochStats.recon = epochStats.recon + reconLossSum
    epochStats.kld = epochStats.kld + kldLossSum
    
    local totalTime = os.clock() - startTime;
    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, totalTime, totalLossSum,
        optimState.learningRate, dataLoadingTime))
        
    print(string.format('  Reconstruction loss: %f', reconLossSum))
    print(string.format('  KLD loss: %f', kldLossSum))

    trainLogger:add{
        ['Total Loss'] = totalLossSum,
        ['Reconstruction Loss'] = reconLossSum,
        ['KLD Loss'] = kldLossSum
    }
    
    totalBatchCount = totalBatchCount + 1
end

-- train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
local function train(model, imgLoader, opt, epoch)
    -- Initialize logging stuff
    if trainLogger == nil then
        trainLogger = optim.Logger(paths.concat(opt.outDir, 'train.log'))
        -- Put symlink to this log in the root directory
        os.execute('ln -sf ' .. opt.outDir .. 'train.log train.log')
    end
    batchNumber = 0

    -- save model
    --this should happen at the end of training, but we keep breaking save so I put it first.
    collectgarbage()
    
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)

    local params, newRegime = paramsForEpoch(epoch)
    if newRegime then
        optimState = {
        learningRate = params.learningRate,
        weightDecay = params.weightDecay
        }
    end
    cutorch.synchronize()

    -- set the dropouts to training mode
    model.trainNet:training()
    
    local tm = torch.Timer()
    
    epochStats.total = 0
    epochStats.recon = 0
    epochStats.kld = 0
    
    for i = 1, opt.epochSize do
        trainSuperBatch(model, imgLoader, opt, epoch)
    end
    
    cutorch.synchronize()

    local scaleFactor = 1.0 / (opt.batchSize * opt.superBatches * opt.epochSize)
    epochStats.total = epochStats.total * scaleFactor
    epochStats.recon = epochStats.recon * scaleFactor
    epochStats.kld = epochStats.kld * scaleFactor
    
    -- trainLogger:add{
    --     ['total loss (train set)'] = epochStats.total,
    --     ['guide loss (train set)'] = epochStats.recon,
    --     ['kld loss (train set)'] = epochStats.kld
    -- }
    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
        .. 'average loss (per batch): %.2f \t '
        .. 'accuracy(%%):\t top-1 %.2f\t',
        epoch, tm:time().real, epochStats.total, epochStats.total))
    print('\n')
end

-------------------------------------------------------------------------------------------


return train
