
require('nnModules')

local useResidualBlock = true
local useBatchNorm = true

local function addConvElement(network,iChannels,oChannels,size,stride,padding)
    network:add(cudnn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    if useBatchNorm then network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3)) end
    network:add(cudnn.ReLU(true))
end

local function addLinearElement(network,iChannels,oChannels)
    network:add(nn.Linear(iChannels, oChannels))
    if useBatchNorm then network:add(cudnn.BatchNormalization(oChannels, 1e-3)) end
    network:add(cudnn.ReLU(true))
end

local function addUpConvElement(network,iChannels,oChannels,size,stride,padding,extra)
    network:add(cudnn.SpatialFullConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding,extra,extra))
    --network:add(nn.SpatialUpSamplingNearest(stride))
    --network:add(nn.SpatialConvolution(iChannels,oChannels,size,size,1,1,padding,padding))
    if useBatchNorm then network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3)) end
    network:add(cudnn.ReLU(true))
end

local function addResidualBlock(network,iChannels,oChannels,size,stride,padding)
    --addConvElement(network,iChannels,oChannels,size,stride,padding)
    --addConvElement(network,iChannels,oChannels,size,stride,padding)

    local s = nn.Sequential()
        
    s:add(cudnn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    if useBatchNorm then s:add(cudnn.SpatialBatchNormalization(oChannels,1e-3)) end
    s:add(cudnn.ReLU(true))
    s:add(cudnn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    if useBatchNorm then s:add(cudnn.SpatialBatchNormalization(oChannels,1e-3)) end
    
    if useResidualBlock then
        --local shortcut = nn.narrow(3, )
        
        local block = nn.Sequential()
            :add(nn.ConcatTable()
            :add(s)
            :add(nn.Identity()))
            :add(nn.CAddTable(true))
        network:add(block)
    else
        s:add(nn.ReLU(true))
        network:add(s)
    end
end

local function createVGG(opt)
    local contentBatch = torch.FloatTensor(opt.batchSize, 3, opt.cropSize, opt.cropSize)
    
    local vggIn = loadcaffe.load('models/VGG_ILSVRC_19_layers_deploy.prototxt',
                                 'models/VGG_ILSVRC_19_layers.caffemodel', 'nn'):float()
    local vggContentOut = nn.Sequential()
    
    local contentDepth = 9
    
    local contentName = 'relu2_2'
    
    for i = 1, contentDepth do
        local layer = vggIn:get(i)
        local name = layer.name
        --print('layer ' .. i .. ': ' .. name)
        local layerType = torch.type(layer)
        
        vggContentOut:add(layer)
    end
    
    vggIn = nil
    collectgarbage()
    return vggContentOut
end

local function createEncoder(opt)
    local encoder = nn.Sequential()

    addConvElement(encoder, 1, 64, 3, 2, 1)
    addConvElement(encoder, 64, 128, 3, 1, 1)
    
    addConvElement(encoder, 128, 128, 3, 2, 1)
    addConvElement(encoder, 128, 256, 3, 1, 1)
    
    addConvElement(encoder, 256, 256, 3, 2, 1)
    addConvElement(encoder, 256, 256, 3, 1, 1)
    
    --addResidualBlock(encoder, 256, 256, 3, 1, 1)

    return encoder
end

local function createMidLevel(opt)
    local encoder = nn.Sequential()

    addResidualBlock(encoder, 256, 256, 3, 1, 1)

    return encoder
end

local function createGlobalLevel(opt)
    local globalLevel = nn.Sequential()

    addConvElement(globalLevel, 256, 256, 3, 2, 1)
    --addResidualBlock(globalLevel, 256, 256, 3, 1, 1)
    
    addConvElement(globalLevel, 256, 256, 3, 2, 1)
    --addResidualBlock(globalLevel, 256, 256, 3, 1, 1)

    globalLevel:add(nn.Reshape(12544, true))
    addLinearElement(globalLevel, 12544, 1024)
    addLinearElement(globalLevel, 1024, 512)
    
    return globalLevel
end

local function createGlobalToFusion(opt)
    local globalToFusion = nn.Sequential()

    addLinearElement(globalToFusion, 512, 256)
    globalToFusion:add(nn.Replicate(28, 3))
    globalToFusion:add(nn.Replicate(28, 4))
    
    return globalToFusion
end

local function createDecoder(opt)
    local decoder = nn.Sequential()

    addConvElement(decoder, 512, 256, 3, 1, 1)
    addConvElement(decoder, 256, 128, 3, 1, 1)
    --addResidualBlock(decoder, 128, 128, 3, 1, 1)

    addUpConvElement(decoder, 128, 64, 3, 2, 1, 1)
    addConvElement(decoder, 64, 64, 3, 1, 1)
    
    addUpConvElement(decoder, 64, 32, 3, 2, 1, 1)
    
    return decoder
end

local function createDecoderToRGB(opt)
    local decoderToRGB = nn.Sequential()
    decoderToRGB:add(nn.SpatialConvolution(32, 3, 3, 3, 1, 1, 1, 1))
    if opt.TVWeight > 0 then
        print('adding RGB TV loss')
        local tvModule = nn.TVLoss(opt.TVWeight, opt.batchSize):float()
        tvModule:cuda()
        decoderToRGB:add(tvModule)
    end
    return decoderToRGB
end

local function createDecoderToLAB(opt)
    local decoderToLAB = nn.Sequential()
    decoderToLAB:add(nn.SpatialConvolution(32, 3, 3, 3, 1, 1, 1, 1))
    if opt.TVWeight > 0 then
        print('adding LAB TV loss')
        local tvModule = nn.TVLoss(opt.TVWeight, opt.batchSize):float()
        tvModule:cuda()
        decoderToLAB:add(tvModule)
    end
    return decoderToLAB
end

local function createParamPredictor(opt)
    -- Turn encoder output into two equal sized blocks
    -- First is mean, second is log(stddev^2)
    -- TODO: Slap some more residual blocks on each of these, to make them
    --    less correlated?
    local pp = nn.ConcatTable()
    pp:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    pp:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    return pp
end

local function createReparameterizer(opt)
    -- Input is {{mu, logSigmaSq}, randomness}
    local params = nn.Identity()()
    local randomness = nn.Identity()()

    -- Extract mu and logSigmaSq from the input table
    local mu = nn.SelectTable(1)(params)
    local logSigmaSq = nn.SelectTable(2)(params)

    -- Compute sigma (log(sigma^2) = 2log(sigma))
    local sigma = nn.Exp()(nn.MulConstant(0.5, true)(logSigmaSq))

    -- Shift and scale the randomness
    local output = nn.CAddTable()({mu, nn.CMulTable()({sigma, randomness})})

    return nn.gModule({params, randomness}, {output})
end

local function createClassifier(opt)
    local classificationNet = nn.Sequential()
    classificationNet:add(nn.Linear(512, 256))
    classificationNet:add(nn.ReLU(true))
    classificationNet:add(nn.Linear(256, 205))
    classificationNet:add(nn.ReLU(true))
    return classificationNet
end

local function createPredictionNet(opt, subnets)
    -- Input nodes
    local grayscaleImage = nn.Identity()():annotate({name = 'grayscaleImage'})
    local randomness = nn.Identity()():annotate({name = 'randomness'})

    -- Intermediates
    local encoderOutput = subnets.encoder(grayscaleImage):annotate({name = 'encoderOutput'})
    local midLevelOutput = subnets.midLevel(encoderOutput):annotate({name = 'midLevelOutput'})
    local globalLevelOutput = subnets.globalLevel(encoderOutput):annotate({name = 'globalLevelOutput'})
    local globalToFusionOutput = subnets.globalToFusion(globalLevelOutput):annotate({name = 'globalToFusionOutput'})
    local fusionOutput = nn.JoinTable(1, 3)({midLevelOutput, globalToFusionOutput}):annotate({name = 'fusionOutput'})
    local params = subnets.paramPredictor(fusionOutput):annotate({name = 'params'})
    local sample = subnets.reparameterizer({params, randomness}):annotate({name = 'sample'})
    local decoderOutput = subnets.decoder(sample):annotate({name = 'decoderOutput'})
    local RGBOutput = subnets.decoderToRGB(decoderOutput):annotate({name = 'RGBOutput'})
    --local LABOutput = subnets.decoderToLAB(decoderOutput):annotate({name = 'LABOutput'})

    local predictionNet = nn.gModule({grayscaleImage, randomness}, {RGBOutput})
    cudnn.convert(predictionNet, cudnn)
    predictionNet = predictionNet:cuda()
    return predictionNet
end

local function createTrainingNet(opt, subnets)
    -- Input nodes
    local grayscaleImage = nn.Identity()():annotate({name = 'grayscaleImage'})
    local randomness = nn.Identity()():annotate({name = 'randomness'})
    local colorImage = nn.Identity()():annotate({name = 'colorImage'})
    local targetContent = nn.Identity()():annotate({name = 'targetContent'})
    local targetCategories = nn.Identity()():annotate({name = 'targetCategories'}) 

    -- Intermediates
    local encoderOutput = subnets.encoder(grayscaleImage):annotate({name = 'encoderOutput'})
    local midLevelOutput = subnets.midLevel(encoderOutput):annotate({name = 'midLevelOutput'})
    local globalLevelOutput = subnets.globalLevel(encoderOutput):annotate({name = 'globalLevelOutput'})
    local classProbabilitiesPreLog = subnets.classifier(globalLevelOutput):annotate({name = 'classProbabilitiesPreLog'})
    local globalToFusionOutput = subnets.globalToFusion(globalLevelOutput):annotate({name = 'globalToFusionOutput'})
    local fusionOutput = nn.JoinTable(1, 3)({midLevelOutput, globalToFusionOutput}):annotate({name = 'fusionOutput'})
    local params = subnets.paramPredictor(fusionOutput):annotate({name = 'params'})
    local sample = subnets.reparameterizer({params, randomness}):annotate({name = 'sample'})
    local decoderOutput = subnets.decoder(sample):annotate({name = 'decoderOutput'})
    local RGBOutput = subnets.decoderToRGB(decoderOutput):annotate({name = 'RGBOutput'})
    --local LABOutput = subnets.decoderToLAB(decoderOutput):annotate({name = 'LABOutput'})
    
    -- Losses
    
    print('adding class loss')
    local classProbabilities = cudnn.LogSoftMax()(classProbabilitiesPreLog):annotate({name = 'classProbabilities'})
    local classLoss = nn.ClassNLLCriterion()({classProbabilities, targetCategories}):annotate{name = 'classLoss'}
    --local classLoss = nn.CrossEntropyCriterion()({classProbabilitiesPreLog, targetCategories}):annotate({name = 'classLoss'})
    
    print('adding pixel RGB loss')
    local pixelRGBLoss = nn.MSECriterion()({RGBOutput, colorImage}):annotate({name = 'pixelLoss'})

    print('adding content loss')
    local perceptualContent = subnets.vggNet(RGBOutput):annotate({name = 'perceptualContent'})
    local contentLoss = nn.MSECriterion()({perceptualContent, targetContent}):annotate({name = 'contentLoss'})

    print('adding KLD loss')
    local kldLoss = nn.KLDCriterion()({params}):annotate({name = 'kldLoss'})
    
    local classLosMul = nn.MulConstant(opt.classWeight, true)(classLoss)
    local pixelRGBLossMul = nn.MulConstant(opt.pixelRGBWeight, true)(pixelRGBLoss)
    --local pixelLABLossMul = nn.MulConstant(opt.pixelLABWeight, true)(pixelLABLoss)
    local contentLossMul = nn.MulConstant(opt.contentWeight, true)(contentLoss)
    local kldLossMul = nn.MulConstant(opt.KLDWeight, true)(kldLoss)

    -- Full training network including all loss functions
    local trainingNet = nn.gModule({grayscaleImage, randomness, colorImage, targetContent, targetCategories},
        {classLosMul, pixelRGBLossMul, contentLossMul, kldLossMul})

    cudnn.convert(trainingNet, cudnn)
    trainingNet = trainingNet:cuda()
    graph.dot(trainingNet.fg, 'graphForward', 'graphForward')
    graph.dot(trainingNet.bg, 'graphBackward', 'graphBackward')
    return trainingNet, classProbabilities
end

local function createModel(opt)
    print('Creating model')

    -- Return table
    local r = {}

    -- Create individual sub-networks
    local subnets = {
        encoder = createEncoder(opt),
        midLevel = createMidLevel(opt),
        globalLevel = createGlobalLevel(opt),
        classifier = createClassifier(opt),
        globalToFusion = createGlobalToFusion(opt),
        paramPredictor = createParamPredictor(opt),
        reparameterizer = createReparameterizer(opt),
        decoder = createDecoder(opt),
        decoderToRGB = createDecoderToRGB(opt),
        decoderToLAB = createDecoderToLAB(opt),
        vggNet = createVGG(opt)
    }
    r.encoder = subnets.encoder
    r.decoder = subnets.decoder
    r.classifier = subnets.classifier -- Needs to have their intermediates cleared for saving.
    r.vggNet = subnets.vggNet  -- Needs to be exposed to gradients be zeroed

    -- Create composite nets
    r.predictionNet = createPredictionNet(opt, subnets)
    r.trainingNet, r.classProbabilities = createTrainingNet(opt, subnets)
    
    return r
end


return {
    createModel = createModel
}
