local imageLoader = require('imageLoader')
local torchUtil = require('torchUtil')

local describeNets = true

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
        {  2,     3,   5e-4,   0 },
        {  4,     10,   1e-4,   0 },
        { 11,     20,   5e-5,   0 },
        { 21,     30,   2e-5,   0 },
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
local lossEpoch


-- GPU inputs (preallocate)
local grayscaleInputs = torch.CudaTensor()
local colorTargets = torch.CudaTensor()
local classLabels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
local function trainBatchGraph(model, grayscaleInputsCPU, colorTargetsCPU, classLabelsCPU, opt, epoch)
    local parametersGraph, gradParametersGraph = model.graph:getParameters()
    
    cutorch.synchronize()
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()

    -- transfer over to GPU
    grayscaleInputs:resize(grayscaleInputsCPU:size()):copy(grayscaleInputsCPU)
    colorTargets:resize(colorTargetsCPU:size()):copy(colorTargetsCPU)
    
    --[[local classLabelsCPU = torch.IntTensor(opt.batchSize, 1)
    for i = 1, opt.batchSize do
        classLabelsCPU[i][1] = 1
    end]]
    classLabels:resize(classLabelsCPU:size()):copy(classLabelsCPU)

    if totalBatchCount % 100 == 0 then
        local inClone = colorTargets[1]:clone()
        --inClone:add(0.5)
        inClone = torchUtil.caffeDeprocess(inClone)
        
        local outClone = model.upConvNet:forward(model.downConvNet:forward(grayscaleInputs))[1]:clone()
        outClone = torchUtil.caffeDeprocess(outClone)
        
        image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_in.jpg', inClone)
        image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_out.jpg', outClone)
    end
    
    --local g = model.graph:listModules()
    --print(g)
    
    local classLoss, pixelLoss, contentLoss, totalLoss
    local feval = function(x)
        local contentTargets = model.vggNet:forward(colorTargets):clone()
        
        model.graph:zeroGradParameters()
        
        --print(model.graph)
        local outputLoss = model.graph:forward({grayscaleInputs, colorTargets, contentTargets, classLabels})
        
        classLoss = outputLoss[1][1]
        pixelLoss = outputLoss[2][1]
        contentLoss = outputLoss[3][1]
        
        model.graph:backward({grayscaleInputs, colorTargets, contentTargets, classLabels}, outputLoss)
        
        totalLoss = classLoss + pixelLoss + contentLoss
        
        model.vggNet:zeroGradParameters()
        
        return totalLoss, gradParametersGraph
    end
    optim.adam(feval, parametersGraph, optimState)

    cutorch.synchronize()
    batchNumber = batchNumber + 1
    lossEpoch = lossEpoch + totalLoss

    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, totalLoss,
        optimState.learningRate, dataLoadingTime))
        
    print(string.format('  Class loss: %f', classLoss))
    print(string.format('  Pixel loss: %f', pixelLoss))
    print(string.format('  Content loss: %f', contentLoss))
    
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
    model.graph:training()

    local tm = torch.Timer()
    lossEpoch = 0
    for i = 1, opt.epochSize do
        local batch = imageLoader.sampleBatch(imgLoader)
        trainBatchGraph(model, batch.grayscaleInputs, batch.colorTargets, batch.classLabels, opt, epoch)
    end
    
    cutorch.synchronize()

    lossEpoch = lossEpoch / (opt.batchSize * opt.epochSize)

    trainLogger:add{
    ['avg loss (train set)'] = lossEpoch
    }
    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
        .. 'average loss (per batch): %.2f \t '
        .. 'accuracy(%%):\t top-1 %.2f\t',
        epoch, tm:time().real, lossEpoch, lossEpoch))
    print('\n')

    -- save model
    collectgarbage()

    -- clear the intermediate states in the model before saving to disk
    -- this saves lots of disk space
    transformNetwork:clearState()
    
    torch.save(opt.outDir .. 'models/transform' .. epoch .. '.t7', transformNetwork)
end

-------------------------------------------------------------------------------------------


return train
