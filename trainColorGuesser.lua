
local imageLoader = require('imageLoader')
local torchUtil = require('torchUtil')

local debugBatchIndices = {[500]=true, [5000]=true, [20000]=true}
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
        {  2,     2,   1e-4,   0 },
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

local timer = torch.Timer()
local dataTimer = torch.Timer()

local function toCPUTensor(t)
    return torch.FloatTensor(t:size()):copy(t)
end

local function yuv2lab(iCPU)
    --local iCPU = toCPUTensor(i)
    return image.rgb2lab( image.yuv2rgb(iCPU) )
end

local function lab2yuv(iCPU)
    --local iCPU = toCPUTensor(i)
    return image.rgb2yuv( image.lab2rgb(iCPU) )
end

local function predictionCorrectedRGB(YImageGPU, RGBImageGPU)
    local YImage = toCPUTensor(YImageGPU)
    local RGBImage = toCPUTensor(RGBImageGPU)
    
    YImage:add(0.5)
    local YRepeated = torch.repeatTensor( YImage, 3, 1, 1 )
    YRepeated[2]:zero()
    YRepeated[3]:zero()
    local luminance = yuv2lab(YRepeated)

    local LABImage = image.rgb2lab(RGBImage)
    local LABImage = image.scale( LABImage, YImage:size()[2], YImage:size()[3] )
    
    LABImage[1] = luminance[1]
    return image.lab2rgb( LABImage )
end

-- 4. trainSuperBatch - Used by train() to train a superbatch.
local function trainColorGuesserSuperBatch(model, imgLoader, opt, epoch)
    
    local parameters, gradParameters = model.colorGuesserNet:getParameters()
    
    cutorch.synchronize()

    local dataLoadingTime = 0
    timer:reset()

    local guideLossSum, totalLossSum = 0, 0
    local feval = function(x)
        model.colorGuesserNet:zeroGradParameters()
        
        for superBatch = 1, opt.superBatches do
            local loadTimeStart = dataTimer:time().real
            local batch = imageLoader.sampleBatch(imgLoader)
            local loadTimeEnd = dataTimer:time().real
            dataLoadingTime = dataLoadingTime + (loadTimeEnd - loadTimeStart)
            
            -- transfer over to GPU
            grayscaleInputs:resize(batch.grayscaleInputs:size()):copy(batch.grayscaleInputs)
            RGBTargets:resize(batch.RGBTargets:size()):copy(batch.RGBTargets)
            
            model.colorEncoder:evaluate()
            local colorGuideTargets = model.colorEncoder:forward(RGBTargets):clone()
            model.colorEncoder:training()
            
            local outputLoss = model.colorGuesserNet:forward({grayscaleInputs, colorGuideTargets})
            
            local guideLoss = outputLoss[1]
            
            guideLossSum = guideLossSum + guideLoss
            totalLossSum = totalLossSum + guideLoss
            
            model.colorGuesserNet:backward({grayscaleInputs, colorGuideTargets}, outputLoss)
            
            if superBatch == 1 and debugBatchIndices[totalBatchCount] then
                torchUtil.dumpGraph(model.colorGuesserNet, opt.outDir .. 'graphDump' .. totalBatchCount .. '.csv')
            end

            -- Output test samples
            if superBatch == 1 and totalBatchCount % 100 == 0 then
            
                model.finalColorizerNet:evaluate()
                model.colorGuidePredictionNet:evaluate()
            
                -- Save ground truth RGB image
                local inClone = RGBTargets[1]:clone()
                inClone = torchUtil.caffeDeprocess(inClone)
                image.save(opt.outDir .. 'samples/iter' .. totalBatchCount .. '_groundTruth.jpg', inClone)
                
                collectgarbage()
                
                local predictedGuidePrediction = model.finalColorizerNet:forward(grayscaleInputs)
                
                local predictedGuideRGB = torchUtil.caffeDeprocess(predictedGuidePrediction[1]:clone())
                image.save(opt.outDir .. 'samples/iter' .. totalBatchCount .. '_predictedGuideRGBSmall.jpg', predictedGuideRGB)
                
                local predictedGuideRGBCorrected = predictionCorrectedRGB(grayscaleInputs[1], predictedGuideRGB)
                image.save(opt.outDir .. 'samples/iter' .. totalBatchCount .. '_predictedGuideRGBBig.jpg', predictedGuideRGBCorrected)
                
                
                local trueGuidePrediction = model.colorGuidePredictionNet:forward({grayscaleInputs, RGBTargets})
                
                local trueGuideRGB = torchUtil.caffeDeprocess(trueGuidePrediction[1]:clone())
                image.save(opt.outDir .. 'samples/iter' .. totalBatchCount .. '_trueGuideRGBSmall.jpg', trueGuideRGB)
                
                local trueGuideRGBCorrected = predictionCorrectedRGB(grayscaleInputs[1], trueGuideRGB)
                image.save(opt.outDir .. 'samples/iter' .. totalBatchCount .. '_trueGuideRGBBig.jpg', trueGuideRGBCorrected)
                
                model.finalColorizerNet:training()
                model.colorGuidePredictionNet:training()
            end
        end
        
        model.vggNet:zeroGradParameters()
        
        return totalLossSum, gradParameters
    end
    optim.adam(feval, parameters, optimState)

    cutorch.synchronize()
    batchNumber = batchNumber + 1
    
    epochStats.total = epochStats.total + totalLossSum
    epochStats.guide = epochStats.guide + guideLossSum
    
    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, totalLossSum,
        optimState.learningRate, dataLoadingTime))
        
    print(string.format('  Guide loss: %f', guideLossSum))
    
    dataTimer:reset()
    totalBatchCount = totalBatchCount + 1
end

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
    model.grayEncoder:clearState()
    model.colorEncoder:clearState()
    model.guideToFusion:clearState()
    model.decoder:clearState()
    model.vggNet:clearState()
    model.guesserEncoder:clearState()
    
    torch.save(opt.outDir .. 'models/colorGuesser' .. epoch .. '.t7', model.colorGuesserNet)
    
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
    model.colorGuesserNet:training()
    
    local tm = torch.Timer()
    
    epochStats.total = 0
    epochStats.guide = 0
    
    for i = 1, opt.epochSize do
        trainColorGuesserSuperBatch(model, imgLoader, opt, epoch)
    end
    
    cutorch.synchronize()

    local scaleFactor = 1.0 / (opt.batchSize * opt.superBatches * opt.epochSize)
    epochStats.total = epochStats.total * scaleFactor
    epochStats.guide = epochStats.guide * scaleFactor
    
    trainLogger:add{
        ['total loss (train set)'] = epochStats.total,
        ['guide loss (train set)'] = epochStats.guide
    }
    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
        .. 'average loss (per batch): %.2f \t '
        .. 'accuracy(%%):\t top-1 %.2f\t',
        epoch, tm:time().real, epochStats.total, epochStats.total))
    print('\n')
end

-------------------------------------------------------------------------------------------


return train
