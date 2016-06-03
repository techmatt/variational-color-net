
require 'optim'

--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

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
        {  1,     20,   1e-3,   0 }, --5e-4
        { 21,     29,   5e-4,   0 },
        { 30,     43,   2e-4,   0 },
        { 44,     52,   5e-5,   0 },
        { 53,    1e8,   1e-5,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.outDir, 'train.log'))
local batchNumber
local totalBatchCount = 0

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train(imageLoader)
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)

    local params, newRegime = paramsForEpoch(epoch)
    if newRegime then
        optimState = {
        learningRate = params.learningRate,
        weightDecay = params.weightDecay
        }
    end
    batchNumber = 0
    cutorch.synchronize()

    -- set the dropouts to training mode
    fullNetwork:training()

    local tm = torch.Timer()
    lossEpoch = 0
    for i = 1, opt.epochSize do
        local batch = sampleBatch(imageLoader)
        trainBatch(batch.inputs, batch.labels)
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
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = fullNetwork:getParameters()

-- Run it through the network once to get the proper size for the gradient
-- All the gradients will come from the extra loss modules, so we just pass
-- zeros into the top of the net on the backward pass.
local zeroGradOutputs = nil

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
    cutorch.synchronize()
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()

    -- transfer over to GPU
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    labels:resize(labelsCPU:size()):copy(labelsCPU)

    if describeNets and totalBatchCount == 0 then
        describeNet(fullNetwork, inputs, opt.outDir .. 'full/')
        describeNet(vggContentNetwork, labels, opt.outDir .. 'content/')
    end
    
    if not zeroGradOutputs then
        local output = fullNetwork:forward(inputs)
        zeroGradOutputs = inputs.new(#output):zero()
    end

    local loss, contentTargets
    feval = function(x)
        contentTargets = vggContentNetwork:forward(labels):clone()
        contentLossModule.target = contentTargets
        
        if totalBatchCount % 100 == 0 then
            local inClone = labels[1]:clone()
            --inClone:add(0.5)
            inClone = caffeDeprocess(inClone)
            
            local outClone = transformNetwork:forward(inputs)[1]:clone()
            outClone = caffeDeprocess(outClone)
            
            image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_in.png', inClone)
            image.save(opt.outDir .. 'samples/sample' .. totalBatchCount .. '_out.png', outClone)
        end
        
        fullNetwork:zeroGradParameters()
        fullNetwork:forward(inputs)
        fullNetwork:backward(inputs, zeroGradOutputs)
        
        loss = contentLossModule.loss
        
        vggContentNetwork:zeroGradParameters()
        
        return loss, gradParameters
    end
    optim.adam(feval, parameters, optimState)

    cutorch.synchronize()
    batchNumber = batchNumber + 1
    lossEpoch = lossEpoch + loss

    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, loss,
        optimState.learningRate, dataLoadingTime))
        
    --print(string.format('  Content loss: %f', contentLossModule.loss))
    
    dataTimer:reset()
    totalBatchCount = totalBatchCount + 1
end
