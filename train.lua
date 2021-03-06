
local imageLoader = require('imageLoader')
local torchUtil = require('torchUtil')

--local debugBatchIndices = {[1]=true, [100]=true, [200]=true}
-- local debugBatchIndices = {[5]=true}
local debugBatchIndices = {}

-- Setup a reused optimization state (for adam/sgd).
local optimState = {
    learningRate = 0.0
}

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
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
local RGBTargets = torch.CudaTensor()
local classLabels = torch.CudaTensor()
local randomness = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()


-- 4. trainSuperBatch - Used by train() to train a superbatch.
local function trainSuperBatch(model, imgLoader, opt, epoch)
    
    local parameters, gradParameters = model.trainingNet:getParameters()
    
    cutorch.synchronize()

    local dataLoadingTime = 0
    timer:reset()

    local classLossSum, pixelRGBLossSum, contentLossSum, kldLossSum, totalLossSum = 0, 0, 0, 0, 0
    local top1, top5 = 0, 0
    local feval = function(x)
        model.trainingNet:zeroGradParameters()
        
        for superBatch = 1, opt.superBatches do
            local loadTimeStart = dataTimer:time().real
            local batch = imageLoader.sampleBatch(imgLoader)
            local loadTimeEnd = dataTimer:time().real
            dataLoadingTime = dataLoadingTime + (loadTimeEnd - loadTimeStart)
            
            local randomnessCPU
            if opt.useRandomness then
                randomnessCPU = torch.randn(opt.batchSize, 512, 28, 28)
                -- -- [Fully-connected bottleneck version]
                -- randomnessCPU = torch.randn(opt.batchSize, 4704)
            else
                -- one lets it make some use of the sigma terms
                randomnessCPU = torch.FloatTensor(opt.batchSize, 512, 28, 28):zero():add(1.0)
            end
            
            -- transfer over to GPU
            grayscaleInputs:resize(batch.grayscaleInputs:size()):copy(batch.grayscaleInputs)
            RGBTargets:resize(batch.RGBTargets:size()):copy(batch.RGBTargets)
            classLabels:resize(batch.classLabels:size()):copy(batch.classLabels)
            randomness:resize(randomnessCPU:size()):copy(randomnessCPU)
        
            local contentTargets = model.vggNet:forward(RGBTargets):clone()
            
            local outputLoss = model.trainingNet:forward({grayscaleInputs, randomness, RGBTargets, contentTargets, classLabels})
            
            
            local classLoss = outputLoss[1][1]
            local pixelRGBLoss = outputLoss[2][1]
            local contentLoss = outputLoss[3][1]
            local kldLoss = outputLoss[4][1]
            
            classLossSum = classLossSum + classLoss
            pixelRGBLossSum = pixelRGBLossSum + pixelRGBLoss
            contentLossSum = contentLossSum + contentLoss
            kldLossSum = kldLossSum + kldLoss
            totalLossSum = totalLossSum + classLoss + pixelRGBLoss + contentLoss + kldLoss
            -- Check nans
            assert(totalLossSum == totalLossSum, 'NaN in loss!')
            
            local classProbabilities = model.classProbabilities.data.module.output
            
            model.trainingNet:backward({grayscaleInputs, randomness, RGBTargets, contentTargets, classLabels}, outputLoss)
            
            if superBatch == 1 then
                if debugBatchIndices[totalBatchCount] then
                    torchUtil.dumpGraph(model.trainingNet, opt.outDir .. 'graphDump' .. totalBatchCount .. '.csv')
                end
                
                do
                    local _, predictions = classProbabilities:float():sort(2, true) -- descending
                    for b = 1, opt.batchSize do
                        --print(predictions[b][1] .. ' vs ' .. classLabelsCPU[b][1])
                        if predictions[b][1] == batch.classLabels[b] then
                            top1 = top1 + 1
                        end
                        if predictions[b][1] == batch.classLabels[b] or
                           predictions[b][2] == batch.classLabels[b] or
                           predictions[b][3] == batch.classLabels[b] or
                           predictions[b][4] == batch.classLabels[b] or
                           predictions[b][5] == batch.classLabels[b] then
                            top5 = top5 + 1
                        end
                    end
                    top1 = top1 * 100 / opt.batchSize   
                    top5 = top5 * 100 / opt.batchSize
                end
            end

            -- Output test samples
            if superBatch == 1 and totalBatchCount % 100 == 0 then
                -- Copy image #1 into the entire batch for grayscaleInputs
                -- This allows us to output N random samples from the network, where N = batch size
                for batchIndex = 2, opt.batchSize do
                    grayscaleInputs[batchIndex]:copy(grayscaleInputs[1])
                end

                -- Save ground truth RGB image
                local inClone = RGBTargets[1]:clone()
                inClone = torchUtil.caffeDeprocess(inClone)
                image.save(opt.outDir .. 'samples/iter' .. totalBatchCount .. '_groundTruth.jpg', inClone)
                
                -- Save predicted images
                local prediction = model.predictionNet:forward({grayscaleInputs, randomness})
                for testSampleIndex = 1, opt.numTestSamples do
                    local predictionRGB = torchUtil.caffeDeprocess(prediction[testSampleIndex]:clone())
                    
                    image.save(opt.outDir .. 'samples/iter' .. totalBatchCount .. '_sample' .. testSampleIndex .. '_predictedRGBDebug.jpg', predictionRGB)
                    
                    local predictionRGB = torchUtil.predictionCorrectedRGB(grayscaleInputs[1], predictionRGB)
                    
                    image.save(opt.outDir .. 'samples/iter' .. totalBatchCount .. '_sample' .. testSampleIndex .. '_predictedRGB.jpg', predictionRGB)
                end
            end
        end
        
        model.vggNet:zeroGradParameters()
        
        return totalLossSum, gradParameters
    end
    optim.adam(feval, parameters, optimState)

    cutorch.synchronize()
    batchNumber = batchNumber + 1
    
    epochStats.total = epochStats.total + totalLossSum
    epochStats.class = epochStats.class + classLossSum
    epochStats.pixelRGB = epochStats.pixelRGB + pixelRGBLossSum
    epochStats.content = epochStats.content + contentLossSum
    epochStats.kld = epochStats.kld + kldLossSum
    
    epochStats.top1Accuracy = top1
    epochStats.top5Accuracy = top5

    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, totalLossSum,
        optimState.learningRate, dataLoadingTime))
        
    print(string.format('  Top 1 accuracy: %f%%', top1))
    print(string.format('  Top 5 accuracy: %f%%', top5))
    print(string.format('  Class loss: %f', classLossSum))
    print(string.format('  RGB loss: %f', pixelRGBLossSum))
    print(string.format('  Content loss: %f', contentLossSum))
    print(string.format('  KLD loss: %f', kldLossSum))
    
    dataTimer:reset()
    totalBatchCount = totalBatchCount + 1
end

-- 4. trainSuperBatch - Used by train() to train a classifier-only superbatch.
local function trainSuperBatchClassifier(model, imgLoader, opt, epoch)
    local parameters, gradParameters = model.classifierTrainingNet:getParameters()
    
    cutorch.synchronize()

    local dataLoadingTime = 0
    timer:reset()

    local classLossSum = 0
    local top1, top5 = 0, 0
    local feval = function(x)
        model.classifierTrainingNet:zeroGradParameters()
        
        for superBatch = 1, opt.superBatches do
            local loadTimeStart = dataTimer:time().real
            local batch = imageLoader.sampleBatch(imgLoader)
            local loadTimeEnd = dataTimer:time().real
            dataLoadingTime = dataLoadingTime + (loadTimeEnd - loadTimeStart)
            
            -- transfer over to GPU
            grayscaleInputs:resize(batch.grayscaleInputs:size()):copy(batch.grayscaleInputs)
            classLabels:resize(batch.classLabels:size()):copy(batch.classLabels)
            
            local outputLoss = model.classifierTrainingNet:forward({grayscaleInputs, classLabels})
            
            local classLoss = outputLoss[1]
            
            classLossSum = classLossSum + classLoss
            
            local classProbabilities = model.classifierClassProbabilities.data.module.output
            
            model.classifierTrainingNet:backward({grayscaleInputs, classLabels}, outputLoss)
            
            if superBatch == 1 then
                if debugBatchIndices[totalBatchCount] then
                    torchUtil.dumpGraph(model.classifierTrainingNet, opt.outDir .. 'graphDump' .. totalBatchCount .. '.csv')
                end
                
                do
                    local _, predictions = classProbabilities:float():sort(2, true) -- descending
                    for b = 1, opt.batchSize do
                        --print(predictions[b][1] .. ' vs ' .. classLabelsCPU[b][1])
                        if predictions[b][1] == batch.classLabels[b] then
                            top1 = top1 + 1
                        end
                        if predictions[b][1] == batch.classLabels[b] or
                           predictions[b][2] == batch.classLabels[b] or
                           predictions[b][3] == batch.classLabels[b] or
                           predictions[b][4] == batch.classLabels[b] or
                           predictions[b][5] == batch.classLabels[b] then
                            top5 = top5 + 1
                        end
                    end
                    top1 = top1 * 100 / opt.batchSize
                    top5 = top5 * 100 / opt.batchSize
                end
            end
        end
        
        return classLossSum, gradParameters
    end
    optim.adam(feval, parameters, optimState)

    cutorch.synchronize()
    batchNumber = batchNumber + 1
    
    epochStats.class = epochStats.class + classLossSum
    
    epochStats.top1Accuracy = top1
    epochStats.top5Accuracy = top5
    
    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, classLossSum,
        optimState.learningRate, dataLoadingTime))
        
    print(string.format('  Top 1 accuracy: %f%%', top1))
    print(string.format('  Top 5 accuracy: %f%%', top5))
    print(string.format('  Class loss: %f', classLossSum))
    
    dataTimer:reset()
    totalBatchCount = totalBatchCount + 1
end

-------------------------------------------------------------------------------------------


-- train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
local function train(model, imgLoader, opt, epoch)
    -- Initialize logging stuff
    if trainLogger == nil then
        trainLogger = optim.Logger(paths.concat(opt.outDir, 'train.log'))
    end
    batchNumber = 0

    -- save model
    --this should happen at the end of training, but we keep breaking save so I put it first.
    collectgarbage()

    -- clear the intermediate states in the model before saving to disk
    -- this saves lots of disk space
    --model.trainingNet:clearState()
    model.encoder:clearState()
    model.decoder:clearState()
    model.classifier:clearState()
    model.vggNet:clearState()
    
    torch.save(opt.outDir .. 'models/transform' .. epoch .. '.t7', model.trainingNet)
    
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
    model.classifierTrainingNet:training()
    model.trainingNet:training()
    
    local tm = torch.Timer()
    
    epochStats.total = 0
    epochStats.class = 0
    epochStats.pixelRGB = 0
    epochStats.content = 0
    epochStats.kld = 0
    
    for i = 1, opt.epochSize do
        if opt.classifierOnly then
            trainSuperBatchClassifier(model, imgLoader, opt, epoch)
        else
            trainSuperBatch(model, imgLoader, opt, epoch)
        end
    end
    
    cutorch.synchronize()

    local scaleFactor = 1.0 / (opt.batchSize * opt.superBatches * opt.epochSize)
    epochStats.total = epochStats.total * scaleFactor
    epochStats.class = epochStats.class * scaleFactor
    epochStats.pixelRGB = epochStats.pixelRGB * scaleFactor
    epochStats.content = epochStats.content * scaleFactor
    epochStats.kld = epochStats.kld * scaleFactor
    
    trainLogger:add{
        ['total loss (train set)'] = epochStats.total,
        ['class loss (train set)'] = epochStats.class,
        ['RGB loss (train set)'] = epochStats.pixelRGB,
        ['content loss (train set)'] = epochStats.content,
        ['KLD loss (train set)'] = epochStats.kld,
    }
    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
        .. 'average loss (per batch): %.2f \t '
        .. 'accuracy(%%):\t top-1 %.2f\t',
        epoch, tm:time().real, epochStats.total, epochStats.total))
    print('\n')
end

-------------------------------------------------------------------------------------------


return train
