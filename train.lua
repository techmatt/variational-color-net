local imageLoader = require('imageLoader')
local torchUtil = require('torchUtil')

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
        {  1,     1,   1e-2,   0 },
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
local classLabels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
local function trainBatch(model, grayscaleInputsCPU, colorTargetsCPU, classLabelsCPU, opt, epoch)
    
    local parameters, gradParameters = model.trainingNet:getParameters()
    
    cutorch.synchronize()
    
    local dataLoadingTime = dataTimer:time().real
    timer:reset()

    -- transfer over to GPU
    grayscaleInputs:resize(grayscaleInputsCPU:size()):copy(grayscaleInputsCPU)
    colorTargets:resize(colorTargetsCPU:size()):copy(colorTargetsCPU)
    classLabels:resize(classLabelsCPU:size()):copy(classLabelsCPU)

    if totalBatchCount % 100 == 0 then
        local inClone = colorTargets[1]:clone()
        --inClone:add(0.5)
        inClone = torchUtil.caffeDeprocess(inClone)
        
        local outClone = model.predictionNet:forward(grayscaleInputs)[1]:clone()
        outClone = torchUtil.caffeDeprocess(outClone)
        
        image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_in.jpg', inClone)
        image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_out.jpg', outClone)
    end
    
    --local g = model.trainingNet:listModules()
    --print(g)
    
    local classLoss, pixelLoss, contentLoss, totalLoss = 0, 0, 0, 0
    local classProbabilities
    local feval = function(x)
        local contentTargets = model.vggNet:forward(colorTargets):clone()
        
        model.trainingNet:zeroGradParameters()
        
        --print(model.trainingNet)
        --TODO: make this correct, currently opt.superBatches must be 1. we need a new batch each time.
        for superBatch = 1, opt.superBatches do
            local outputLoss = model.trainingNet:forward({grayscaleInputs, colorTargets, contentTargets, classLabels})
            
            classLoss = classLoss + outputLoss[1][1]
            pixelLoss = pixelLoss + outputLoss[2][1]
            contentLoss = contentLoss + outputLoss[3][1]
            classProbabilities = outputLoss[4]
            
            model.trainingNet:backward({grayscaleInputs, colorTargets, contentTargets, classLabels}, outputLoss)
            
            totalLoss = totalLoss + classLoss + pixelLoss + contentLoss
        end
        
        model.vggNet:zeroGradParameters()
        
        return totalLoss, gradParameters
    end
    optim.adam(feval, parameters, optimState)

    cutorch.synchronize()
    batchNumber = batchNumber + 1
    
    epochStats.total = epochStats.total + totalLoss
    epochStats.class = epochStats.class + classLoss
    epochStats.pixel = epochStats.pixel + pixelLoss
    epochStats.content = epochStats.content + contentLoss

    local top1 = 0
    local top5 = 0
    do
        local _, predictions = classProbabilities:float():sort(2, true) -- descending
        for b = 1, opt.batchSize do
            --print(predictions[b][1] .. ' vs ' .. classLabelsCPU[b][1])
            if predictions[b][1] == classLabelsCPU[b][1] then
                top1 = top1 + 1
            end
            if predictions[b][1] == classLabelsCPU[b][1] or
               predictions[b][2] == classLabelsCPU[b][1] or
               predictions[b][3] == classLabelsCPU[b][1] or
               predictions[b][4] == classLabelsCPU[b][1] or
               predictions[b][5] == classLabelsCPU[b][1] then
                top5 = top5 + 1
            end
        end
        top1 = top1 * 100 / opt.batchSize
        top5 = top5 * 100 / opt.batchSize
    end
    
    epochStats.top1Accuracy = top1
    epochStats.top5Accuracy = top5

    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, totalLoss,
        optimState.learningRate, dataLoadingTime))
        
    print(string.format('  Top 1 accuracy: %f%%', top1))
    print(string.format('  Top 5 accuracy: %f%%', top5))
    print(string.format('  Class loss: %f', classLoss))
    print(string.format('  Pixel loss: %f', pixelLoss))
    print(string.format('  Content loss: %f', contentLoss))
    
    dataTimer:reset()
    totalBatchCount = totalBatchCount + 1
end

-- 4. trainSuperBatch - Used by train() to train a superbatch.
local function trainSuperBatch(model, imgLoader, opt, epoch)
    
    local parameters, gradParameters = model.trainingNet:getParameters()
    
    cutorch.synchronize()
    
    local dataLoadingTime = dataTimer:time().real
    timer:reset()

    local classLossSum, pixelLossSum, contentLossSum, totalLossSum = 0, 0, 0, 0
    local top1, top5 = 0, 0
    local feval = function(x)
        model.trainingNet:zeroGradParameters()
        
        for superBatch = 1, opt.superBatches do
            local batch = imageLoader.sampleBatch(imgLoader)
            
            -- transfer over to GPU
            grayscaleInputs:resize(batch.grayscaleInputs:size()):copy(batch.grayscaleInputs)
            colorTargets:resize(batch.colorTargets:size()):copy(batch.colorTargets)
            classLabels:resize(batch.classLabels:size()):copy(batch.classLabels)

            if superBatch == 1 and totalBatchCount % 100 == 0 then
                local inClone = colorTargets[1]:clone()
                inClone = torchUtil.caffeDeprocess(inClone)
                
                local outClone = model.predictionNet:forward(grayscaleInputs)[1]:clone()
                outClone = torchUtil.caffeDeprocess(outClone)
                
                image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_in.jpg', inClone)
                image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_out.jpg', outClone)
            end
        
            local contentTargets = model.vggNet:forward(colorTargets):clone()
            
            local outputLoss = model.trainingNet:forward({grayscaleInputs, colorTargets, contentTargets, classLabels})
            
            local classLoss = outputLoss[1][1]
            local pixelLoss = outputLoss[2][1]
            local contentLoss = outputLoss[3][1]
            
            classLossSum = classLossSum + classLoss
            pixelLossSum = pixelLossSum + pixelLoss
            contentLossSum = contentLossSum + contentLoss
            totalLossSum = totalLossSum + classLoss + pixelLoss + contentLoss
            
            local classProbabilities = model.classProbabilities.data.module.output
            
            model.trainingNet:backward({grayscaleInputs, colorTargets, contentTargets, classLabels}, outputLoss)
            
            if superBatch == 1 then
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
        
        model.vggNet:zeroGradParameters()
        
        return totalLossSum, gradParameters
    end
    optim.adam(feval, parameters, optimState)

    cutorch.synchronize()
    batchNumber = batchNumber + 1
    
    epochStats.total = epochStats.total + totalLossSum
    epochStats.class = epochStats.class + classLossSum
    epochStats.pixel = epochStats.pixel + pixelLossSum
    epochStats.content = epochStats.content + contentLossSum
    
    epochStats.top1Accuracy = top1
    epochStats.top5Accuracy = top5

    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, totalLossSum,
        optimState.learningRate, dataLoadingTime))
        
    print(string.format('  Top 1 accuracy: %f%%', top1))
    print(string.format('  Top 5 accuracy: %f%%', top5))
    print(string.format('  Class loss: %f', classLossSum))
    print(string.format('  Pixel loss: %f', pixelLossSum))
    print(string.format('  Content loss: %f', contentLossSum))
    
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
    model.trainingNet:training()

    local tm = torch.Timer()
    
    epochStats.total = 0
    epochStats.class = 0
    epochStats.pixel = 0
    epochStats.content = 0
    
    for i = 1, opt.epochSize do
        --local batch = imageLoader.sampleBatch(imgLoader)
        --trainBatch(model, batch.grayscaleInputs, batch.colorTargets, batch.classLabels, opt, epoch)
        trainSuperBatch(model, imgLoader, opt, epoch)
    end
    
    cutorch.synchronize()

    local scaleFactor = 1.0 / (opt.batchSize * opt.epochSize)
    epochStats.total = epochStats.total * scaleFactor
    epochStats.class = epochStats.class * scaleFactor
    epochStats.pixel = epochStats.pixel * scaleFactor
    epochStats.content = epochStats.content * scaleFactor
    
    trainLogger:add{
        ['total loss (train set)'] = epochStats.total,
        ['class loss (train set)'] = epochStats.class,
        ['pixel loss (train set)'] = epochStats.pixel,
        ['content loss (train set)'] = epochStats.content,
    }
    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
        .. 'average loss (per batch): %.2f \t '
        .. 'accuracy(%%):\t top-1 %.2f\t',
        epoch, tm:time().real, epochStats.total, epochStats.total))
    print('\n')
end

-------------------------------------------------------------------------------------------


return train
