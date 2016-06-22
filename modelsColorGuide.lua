
require('nnModules')
local torchUtil = require('torchUtil')

local nameLastModParams = torchUtil.nameLastModParams
local transferParams = torchUtil.transferParams

local useResidualBlock = true
local useLeakyReLU = true

local function makeReLU()
    if useLeakyReLU then
        return nn.LeakyReLU(true)
    else
        cudnn.ReLU(true)
    end
end

local function addConvElement(network,iChannels,oChannels,size,stride,padding)
    network:add(cudnn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    nameLastModParams(network)
    network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    nameLastModParams(network)
    network:add(makeReLU())
end

local function addLinearElement(network,iChannels,oChannels)
    network:add(nn.Linear(iChannels, oChannels))
    nameLastModParams(network)
    network:add(cudnn.BatchNormalization(oChannels, 1e-3))
    nameLastModParams(network)
    network:add(makeReLU())
end

local function addLinearTanhElement(network,iChannels,oChannels)
    network:add(nn.Linear(iChannels, oChannels))
    nameLastModParams(network)
    network:add(cudnn.BatchNormalization(oChannels, 1e-3))
    nameLastModParams(network)
    network:add(nn.Tanh())
end

local function addUpConvElement(network,iChannels,oChannels,size,stride,padding,extra)
    network:add(cudnn.SpatialFullConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding,extra,extra))
    nameLastModParams(network)
    --network:add(nn.SpatialUpSamplingNearest(stride))
    --network:add(nn.SpatialConvolution(iChannels,oChannels,size,size,1,1,padding,padding))
    network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    nameLastModParams(network)
    network:add(makeReLU())
end

local function addResidualBlock(network,iChannels,oChannels,size,stride,padding)
    --addConvElement(network,iChannels,oChannels,size,stride,padding)
    --addConvElement(network,iChannels,oChannels,size,stride,padding)

    local s = nn.Sequential()
    s.paramName = network.paramName .. '_ResBlock' .. #(network:listModules())
    
    s:add(cudnn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    nameLastModParams(s)
    s:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    nameLastModParams(s)
    s:add(makeReLU())
    s:add(cudnn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    nameLastModParams(s)
    s:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    nameLastModParams(s)
    
    if useResidualBlock then
        --local shortcut = nn.narrow(3, )
        
        local block = nn.Sequential()
            :add(nn.ConcatTable()
            :add(s)
            :add(nn.Identity()))
            :add(nn.CAddTable(true))
        network:add(block)
    else
        s:add(makeReLU())
        network:add(s)
    end
end

local function createVGG(opt)
    if opt.classifierOnly then return nn.Identity() end
    local contentBatch = torch.FloatTensor(opt.batchSize, 3, opt.cropSize, opt.cropSize)
    
    local vggIn = loadcaffe.load('models/VGG_ILSVRC_19_layers_deploy.prototxt',
                                 'models/VGG_ILSVRC_19_layers.caffemodel', 'nn'):float()
    local vggContentOut = nn.Sequential()
    vggContentOut.paramName = 'VGG'
    
    local contentDepth = 9
    
    local contentName = 'relu2_2'
    
    for i = 1, contentDepth do
        local layer = vggIn:get(i)
        local name = layer.name
        --print('layer ' .. i .. ': ' .. name)
        local layerType = torch.type(layer)
        
        vggContentOut:add(layer)
        nameLastModParams(vggContentOut)
    end
    
    vggIn = nil
    collectgarbage()
    return vggContentOut
end

local function createGrayEncoder(opt)
    local encoder = nn.Sequential()
    encoder.paramName = 'grayEncoder'

    addConvElement(encoder, 1, 64, 3, 2, 1) -- 112
    addConvElement(encoder, 64, 128, 3, 1, 1) -- 112
    
    addConvElement(encoder, 128, 128, 3, 2, 1) -- 56
    addConvElement(encoder, 128, 256, 3, 1, 1) -- 56
    
    addConvElement(encoder, 256, 256, 3, 2, 1) -- 28
    addConvElement(encoder, 256, 256, 3, 1, 1) -- 28

    return encoder
end

local function createColorEncoder(opt)
    local colorEncoder = nn.Sequential()
    colorEncoder.paramName = 'colorEncoder'

    addConvElement(colorEncoder, 3, 32, 3, 1, 1) -- 112
    
    addConvElement(colorEncoder, 32, 64, 3, 2, 1) -- 56
    addConvElement(colorEncoder, 64, 128, 3, 1, 1) -- 56
    
    addConvElement(colorEncoder, 128, 128, 3, 2, 1) -- 28
    addConvElement(colorEncoder, 128, 128, 3, 1, 1) -- 28
    
    addConvElement(colorEncoder, 128, 128, 3, 2, 1) -- 14
    addConvElement(colorEncoder, 128, 128, 3, 1, 1) -- 14
    
    addConvElement(colorEncoder, 128, 128, 3, 2, 1) -- 7
    addConvElement(colorEncoder, 128, 128, 3, 1, 1) -- 7
    
    colorEncoder:add(nn.Reshape(6272, true))
    
    addLinearElement(colorEncoder, 6272, 1024)
    
    colorEncoder:add(nn.Linear(1024, opt.colorGuideSize))
    nameLastModParams(colorEncoder)
    --colorEncoder:add(nn.Tanh())
    
    return colorEncoder
end

local function createGuideToFusion(opt)
    local guideToFusion = nn.Sequential()
    guideToFusion.paramName = 'guideToFusion'

    addLinearElement(guideToFusion, opt.colorGuideSize, 784)
    
    guideToFusion:add(nn.Reshape(16, 7, 7, true))
    
    addUpConvElement(guideToFusion, 16, 32, 3, 2, 1, 1) -- 14
    addConvElement(guideToFusion, 32, 32, 3, 1, 1) -- 14
    
    addUpConvElement(guideToFusion, 32, 64, 3, 2, 1, 1) -- 28

    return guideToFusion
end

local function createDecoder(opt)
    local decoder = nn.Sequential()
    decoder.paramName = 'decoder'

    addConvElement(decoder, 320, 128, 3, 1, 1) -- 28
    --addResidualBlock(decoder, 128, 128, 3, 1, 1)

    addUpConvElement(decoder, 128, 64, 3, 2, 1, 1) -- 56
    addConvElement(decoder, 64, 64, 3, 1, 1) -- 56
    
    addUpConvElement(decoder, 64, 32, 3, 2, 1, 1) -- 112
    
    decoder:add(nn.SpatialConvolution(32, 3, 3, 3, 1, 1, 1, 1))
    nameLastModParams(decoder)
    if opt.TVWeight > 0 then
        print('adding RGB TV loss')
        local tvModule = nn.TVLoss(opt.TVWeight, opt.batchSize):float()
        tvModule:cuda()
        decoder:add(tvModule)
    end
    
    return decoder
end

local function createColorGuideNet(opt, subnets)
    -- Input nodes
    local grayscaleImage = nn.Identity()():annotate({name = 'grayscaleImage'})
    local targetRGB = nn.Identity()():annotate({name = 'targetRGB'})
    local targetContent = nn.Identity()():annotate({name = 'targetContent'})
    
    -- Intermediates
    local grayEncoderOutput = subnets.grayEncoder(grayscaleImage):annotate({name = 'grayEncoderOutput'})
    local colorEncoderOutput = subnets.colorEncoder(targetRGB):annotate({name = 'colorEncoderOutput'})
    local guideToFusionOutput = subnets.guideToFusion(colorEncoderOutput):annotate({name = 'guideToFusionOutput'})
    local fusionOutput = nn.JoinTable(1, 3)({grayEncoderOutput, guideToFusionOutput}):annotate({name = 'fusionOutput'})
    local decoderOutput = subnets.decoder(fusionOutput):annotate({name = 'decoderOutput'})
    
    -- Losses
    
    print('adding pixel RGB loss')
    local pixelRGBLoss = nn.MSECriterion()({decoderOutput, targetRGB}):annotate({name = 'pixelRGBLoss'})
    
    print('adding content loss')
    local perceptualContent = subnets.vggNet(decoderOutput):annotate({name = 'perceptualContent'})
    local contentLoss = nn.MSECriterion()({perceptualContent, targetContent}):annotate({name = 'contentLoss'})

    local pixelRGBLossMul = nn.MulConstant(opt.pixelRGBWeight, true)(pixelRGBLoss)
    local contentLossMul = nn.MulConstant(opt.contentWeight, true)(contentLoss)

    -- Full training network including all loss functions
    local colorGuideNet = nn.gModule({grayscaleImage, targetRGB, targetContent}, {pixelRGBLossMul, contentLossMul})

    cudnn.convert(colorGuideNet, cudnn)
    colorGuideNet = colorGuideNet:cuda()
    graph.dot(colorGuideNet.fg, 'colorGuideForward', 'colorGuideForward')
    --graph.dot(colorGuideNet.bg, 'colorGuideBackward', 'graphBackward')
    return colorGuideNet
end

local function createColorGuidePredictionNet(opt, subnets)
    -- Input nodes
    local grayscaleImage = nn.Identity()():annotate({name = 'grayscaleImage'})
    local targetRGB = nn.Identity()():annotate({name = 'targetRGB'})
    
    -- Intermediates
    local grayEncoderOutput = subnets.grayEncoder(grayscaleImage):annotate({name = 'grayEncoderOutput'})
    local colorEncoderOutput = subnets.colorEncoder(targetRGB):annotate({name = 'colorEncoderOutput'})
    local guideToFusionOutput = subnets.guideToFusion(colorEncoderOutput):annotate({name = 'guideToFusionOutput'})
    local fusionOutput = nn.JoinTable(1, 3)({grayEncoderOutput, guideToFusionOutput}):annotate({name = 'fusionOutput'})
    local decoderOutput = subnets.decoder(fusionOutput):annotate({name = 'decoderOutput'})
    
    -- Full training network including all loss functions
    local colorGuidePredictionNet = nn.gModule({grayscaleImage, targetRGB}, {decoderOutput})

    cudnn.convert(colorGuidePredictionNet, cudnn)
    colorGuidePredictionNet = colorGuidePredictionNet:cuda()
    return colorGuidePredictionNet
end


local function createGuesserEncoder(opt)
    local guesserEncoder = nn.Sequential()
    guesserEncoder.paramName = 'guesserEncoder'

    addConvElement(guesserEncoder, 1, 16, 7, 1, 1) -- 224
    
    addConvElement(guesserEncoder, 16, 32, 3, 2, 1) -- 112
    addConvElement(guesserEncoder, 32, 64, 3, 1, 1) -- 112
    
    addConvElement(guesserEncoder, 64, 128, 3, 2, 1) -- 56
    addConvElement(guesserEncoder, 128, 128, 3, 1, 1) -- 56
    
    addResidualBlock(guesserEncoder, 128, 128, 3, 1, 1)
    addResidualBlock(guesserEncoder, 128, 128, 3, 1, 1)
    addResidualBlock(guesserEncoder, 128, 128, 3, 1, 1)
    
    addConvElement(guesserEncoder, 128, 256, 3, 2, 1) -- 28
    
    addResidualBlock(guesserEncoder, 256, 256, 3, 1, 1)
    addResidualBlock(guesserEncoder, 256, 256, 3, 1, 1)
    addResidualBlock(guesserEncoder, 256, 256, 3, 1, 1)
    
    addConvElement(guesserEncoder, 256, 64, 3, 2, 1) -- 14
    
    guesserEncoder:add(nn.Reshape(12544, true))
    
    --guesserEncoder:add(nn.Dropout(0.5))
    addLinearElement(guesserEncoder, 12544, 2048)
    
    --guesserEncoder:add(nn.Dropout(0.5))
    --addLinearElement(guesserEncoder, 2048, 2048)
    
    guesserEncoder:add(nn.Linear(2048, opt.colorGuideSize))
    nameLastModParams(guesserEncoder)
    
    return guesserEncoder
end

local function craeteDiscriminator(opt)
    local discriminator = nn.Sequential()
    discriminator.paramName = 'discriminator'

    discriminator:add(nn.Dropout(0.5))
    addLinearElement(discriminator, opt.colorGuideSize, 1024)
    discriminator:add(nn.Dropout(0.5))
    addLinearElement(discriminator, 1024, 512)
    discriminator:add(nn.Dropout(0.5))
    addLinearElement(discriminator, 512, 128)
    
    discriminator:add(nn.Linear(128, 2))
    nameLastModParams(discriminator)
    
    discriminator:add(nn.LogSoftMax())
    
    return discriminator
end

-- This version returns a table of {mu, logSigmaSquared} instead
local function createVariationalGuesserEncoder(opt)
    local guesserEncoder = nn.Sequential()
    guesserEncoder.paramName = 'guesserEncoder'

    addConvElement(guesserEncoder, 1, 16, 7, 1, 1) -- 224
    
    addConvElement(guesserEncoder, 16, 32, 3, 2, 1) -- 112
    addConvElement(guesserEncoder, 32, 64, 3, 1, 1) -- 112
    
    addConvElement(guesserEncoder, 64, 128, 3, 2, 1) -- 56
    addConvElement(guesserEncoder, 128, 128, 3, 1, 1) -- 56
    
    addResidualBlock(guesserEncoder, 128, 128, 3, 1, 1)
    addResidualBlock(guesserEncoder, 128, 128, 3, 1, 1)
    addResidualBlock(guesserEncoder, 128, 128, 3, 1, 1)
    
    addConvElement(guesserEncoder, 128, 256, 3, 2, 1) -- 28
    
    addResidualBlock(guesserEncoder, 256, 256, 3, 1, 1)
    addResidualBlock(guesserEncoder, 256, 256, 3, 1, 1)
    addResidualBlock(guesserEncoder, 256, 256, 3, 1, 1)
    
    addConvElement(guesserEncoder, 256, 64, 3, 2, 1) -- 14
    
    guesserEncoder:add(nn.Reshape(12544, true))
    
    addLinearElement(guesserEncoder, 12544, 2048)

    -- guesserEncoder:add(nn.Linear(2048, opt.colorGuideSize))
    -- nameLastModParams(guesserEncoder)

    -- Split into mean and variance
    local split = nn.ConcatTable()
    split.paramName = 'guesserEncoder_splitter'
    split:add(nn.Linear(2048, opt.colorGuideSize))
    nameLastModParams(split)
    split:add(nn.Linear(2048, opt.colorGuideSize))
    nameLastModParams(split)
    guesserEncoder:add(split)
    
    return guesserEncoder
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

-- Take the Gaussian sample and put it through a non-linear transform
-- (so it can model more weirdly-shaped distributions)
local function createSampleTransformer(opt)
    local t = nn.Sequential()
    t.paramName = 'sampleTransformer'

    addLinearTanhElement(t, opt.colorGuideSize, opt.colorGuideSize)
    t:add(nn.Linear(opt.colorGuideSize, opt.colorGuideSize))
    nameLastModParams(t)

    return t

    -- return nn.Identity()
end

local function createDiscriminatorNet(opt, subnets)
    -- Input nodes
    local colorGuides = nn.Identity()():annotate({name = 'colorGuides'})
    local targetCategories = nn.Identity()():annotate({name = 'targetCategories'})
    
    -- Intermediates
    local classProbabilitiesPreLog = subnets.discriminator(colorGuides):annotate({name = 'classProbabilitiesPreLog'})
    
    -- Losses
    print('adding class loss')
    local classProbabilities = cudnn.LogSoftMax()(classProbabilitiesPreLog):annotate({name = 'classProbabilities'})
    local classLoss = nn.ClassNLLCriterion()({classProbabilities, targetCategories}):annotate{name = 'classLoss'}
    
    -- Full training network including all loss functions
    local discriminatorNet = nn.gModule({colorGuides, targetCategories}, {classLoss})

    cudnn.convert(discriminatorNet, cudnn)
    discriminatorNet = discriminatorNet:cuda()
    graph.dot(discriminatorNet.fg, 'discriminatorGraphForward', 'discriminatorGraphForward')
    return discriminatorNet, classProbabilities
end

local function createColorGuesserNet(opt, subnets)
    -- Input nodes
    local grayscaleImage = nn.Identity()():annotate({name = 'grayscaleImage'})
    local targetColorGuide = nn.Identity()():annotate({name = 'targetColorGuide'})
    
    -- Intermediates
    local guesserEncoderOutput = subnets.guesserEncoder(grayscaleImage):annotate({name = 'guesserEncoderOutput'})
    
    -- Losses
    print('adding guide loss')
    local guideLoss = nn.MSECriterion()({guesserEncoderOutput, targetColorGuide}):annotate({name = 'guideLoss'})
    
    -- Full training network including all loss functions
    local colorGuesserNet = nn.gModule({grayscaleImage, targetColorGuide}, {guideLoss})

    cudnn.convert(colorGuesserNet, cudnn)
    colorGuesserNet = colorGuesserNet:cuda()
    graph.dot(colorGuesserNet.fg, 'colorGuesserForward', 'colorGuesserForward')
    --graph.dot(colorGuideNet.bg, 'colorGuideBackward', 'graphBackward')
    return colorGuesserNet, guesserEncoderOutput
end

local function createFinalColorizerNet(opt, subnets)
    -- Input nodes
    local grayscaleImage = nn.Identity()():annotate({name = 'grayscaleImage'})
    
    -- Intermediates
    local guesserEncoderOutput = subnets.guesserEncoder(grayscaleImage):annotate({name = 'guesserEncoderOutput'})
    local guideToFusionOutput = subnets.guideToFusion(guesserEncoderOutput):annotate({name = 'guideToFusionOutput'})
    local grayEncoderOutput = subnets.grayEncoder(grayscaleImage):annotate({name = 'grayEncoderOutput'})
    local fusionOutput = nn.JoinTable(1, 3)({grayEncoderOutput, guideToFusionOutput}):annotate({name = 'fusionOutput'})
    local decoderOutput = subnets.decoder(fusionOutput):annotate({name = 'decoderOutput'})
    
    -- Full training network including all loss functions
    local finalColorizerNet = nn.gModule({grayscaleImage}, {decoderOutput})

    cudnn.convert(finalColorizerNet, cudnn)
    finalColorizerNet = finalColorizerNet:cuda()
    graph.dot(finalColorizerNet.fg, 'finalColorizerForward', 'finalColorizerForward')
    --graph.dot(colorGuideNet.bg, 'colorGuideBackward', 'graphBackward')
    return finalColorizerNet
end

local function createVariationalColorGuesserNet(opt, subnets)
    -- Input nodes
    local grayscaleImage = nn.Identity()():annotate({name = 'grayscaleImage'})
    local randomness = nn.Identity()():annotate({name = 'randomness'})
    local targetColorGuide = nn.Identity()():annotate({name = 'targetColorGuide'})
    
    -- Intermediates
    local predictedParams = subnets.guesserEncoder(grayscaleImage):annotate({name = 'predictedParams'})
    local sample = subnets.reparameterizer({predictedParams, randomness}):annotate({name = 'sample'})
    local xformedSample = subnets.sampleTransformer(sample):annotate({name = 'transformedSample'})
    
    -- Losses
    print('adding guide loss')
    local guideLoss = nn.MSECriterion()({xformedSample, targetColorGuide}):annotate({name = 'guideLoss'})

    print('adding KLD loss')
    local kldLoss = nn.KLDCriterion()(predictedParams):annotate({name = 'kldLoss'})
    kldLoss = nn.MulConstant(opt.KLDWeight, true)(kldLoss)
    
    -- Full training network including all loss functions
    local colorGuesserNet = nn.gModule({grayscaleImage, randomness, targetColorGuide}, {guideLoss, kldLoss})

    cudnn.convert(colorGuesserNet, cudnn)
    colorGuesserNet = colorGuesserNet:cuda()
    graph.dot(colorGuesserNet.fg, 'colorGuesserForward', 'colorGuesserForward')
    --graph.dot(colorGuideNet.bg, 'colorGuideBackward', 'graphBackward')
    return colorGuesserNet
end

local function createVariationalFinalColorizerNet(opt, subnets)
    -- Input nodes
    local grayscaleImage = nn.Identity()():annotate({name = 'grayscaleImage'})
    local randomness = nn.Identity()():annotate({name = 'randomness'})
    
    -- Intermediates
    local predictedParams = subnets.guesserEncoder(grayscaleImage):annotate({name = 'predictedParams'})
    local sample = subnets.reparameterizer({predictedParams, randomness}):annotate({name = 'sample'})
    local xformedSample = subnets.sampleTransformer(sample):annotate({name = 'transformedSample'})
    local guideToFusionOutput = subnets.guideToFusion(xformedSample):annotate({name = 'guideToFusionOutput'})
    local grayEncoderOutput = subnets.grayEncoder(grayscaleImage):annotate({name = 'grayEncoderOutput'})
    local fusionOutput = nn.JoinTable(1, 3)({grayEncoderOutput, guideToFusionOutput}):annotate({name = 'fusionOutput'})
    local decoderOutput = subnets.decoder(fusionOutput):annotate({name = 'decoderOutput'})
    
    -- Full training network including all loss functions
    local finalColorizerNet = nn.gModule({grayscaleImage, randomness}, {decoderOutput})

    cudnn.convert(finalColorizerNet, cudnn)
    finalColorizerNet = finalColorizerNet:cuda()
    graph.dot(finalColorizerNet.fg, 'finalColorizerForward', 'finalColorizerForward')
    --graph.dot(colorGuideNet.bg, 'colorGuideBackward', 'graphBackward')
    return finalColorizerNet
end

local function createModel(opt)
    print('Creating model')

    -- Return table
    local r = {}

    -- Create individual sub-networks
    local subnets = {
        grayEncoder = createGrayEncoder(opt),
        colorEncoder = createColorEncoder(opt),
        guideToFusion = createGuideToFusion(opt),
        discriminator = craeteDiscriminator(opt),
        decoder = createDecoder(opt),
        guesserEncoder = createGuesserEncoder(opt),
        vggNet = createVGG(opt)
    }
    r.grayEncoder = subnets.grayEncoder
    r.colorEncoder = subnets.colorEncoder
    r.guideToFusion = subnets.guideToFusion
    r.decoder = subnets.decoder
    r.guesserEncoder = subnets.guesserEncoder
    r.vggNet = subnets.vggNet  -- Needs to be exposed to gradients be zeroed

    -- Create composite nets
    r.colorGuideNet = createColorGuideNet(opt, subnets)
    r.colorGuidePredictionNet = createColorGuidePredictionNet(opt, subnets)
    r.discriminatorNet, r.discriminatorProbabilities = createDiscriminatorNet(opt, subnets)
    
    r.colorGuesserNet, r.predictedColorGuide = createColorGuesserNet(opt, subnets)
    r.finalColorizerNet = createFinalColorizerNet(opt, subnets)
    
    local pretrainedColorGuide = torch.load('pretrainedModels/colorGuide' .. opt.colorGuideSize .. '.t7')
    pretrainedColorGuide:clearState()
    transferParams(pretrainedColorGuide, r.colorGuideNet)
    transferParams(pretrainedColorGuide, r.colorGuidePredictionNet)
    transferParams(pretrainedColorGuide, r.finalColorizerNet)
        
    return r
end

local function createVariationalModel(opt)
    print('Creating variational model')

    -- Return table
    local r = {}

    -- Create individual sub-networks
    local subnets = {
        grayEncoder = createGrayEncoder(opt),
        colorEncoder = createColorEncoder(opt),
        guideToFusion = createGuideToFusion(opt),
        decoder = createDecoder(opt),
        guesserEncoder = createVariationalGuesserEncoder(opt),
        reparameterizer = createReparameterizer(opt),
        sampleTransformer = createSampleTransformer(opt),
        vggNet = createVGG(opt)
    }
    r.grayEncoder = subnets.grayEncoder
    r.colorEncoder = subnets.colorEncoder
    r.guideToFusion = subnets.guideToFusion
    r.decoder = subnets.decoder
    r.guesserEncoder = subnets.guesserEncoder
    r.vggNet = subnets.vggNet  -- Needs to be exposed to gradients be zeroed

    -- Create composite nets
    r.colorGuideNet = createColorGuideNet(opt, subnets)
    r.colorGuidePredictionNet = createColorGuidePredictionNet(opt, subnets)
    
    r.colorGuesserNet = createVariationalColorGuesserNet(opt, subnets)
    r.finalColorizerNet = createVariationalFinalColorizerNet(opt, subnets)
    
    local pretrainedColorGuide = torch.load('pretrainedModels/colorGuide' .. opt.colorGuideSize .. '.t7')
    pretrainedColorGuide:clearState()
    transferParams(pretrainedColorGuide, r.colorGuideNet)
    transferParams(pretrainedColorGuide, r.colorGuidePredictionNet)
    transferParams(pretrainedColorGuide, r.finalColorizerNet)
        
    return r
end


return {
    createModel = createModel,
    createVariationalModel = createVariationalModel
}
