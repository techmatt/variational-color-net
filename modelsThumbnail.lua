
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

local function addLinearElementNoBN(network,iChannels,oChannels)
    network:add(nn.Linear(iChannels, oChannels))
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

local function createThumbnailToFusion(opt)
    local thumbnailToFusion = nn.Sequential()
    thumbnailToFusion.paramName = 'thumbnailToFusion'

    thumbnailToFusion:add(nn.Reshape(opt.thumbnailFeatures, true))
    addLinearElement(thumbnailToFusion, opt.thumbnailFeatures, 3136)
    
    thumbnailToFusion:add(nn.Reshape(64, 7, 7, true))
    
    addUpConvElement(thumbnailToFusion, 64, 64, 3, 2, 1, 1) -- 14
    addConvElement(thumbnailToFusion, 64, 64, 3, 1, 1) -- 14
    
    addUpConvElement(thumbnailToFusion, 64, 64, 3, 2, 1, 1) -- 28

    return thumbnailToFusion
end

local function createFusionToRGB(opt)
    local fusionToRGB = nn.Sequential()
    fusionToRGB.paramName = 'decoder'

    addConvElement(fusionToRGB, 320, 128, 3, 1, 1) -- 28
    
    addUpConvElement(fusionToRGB, 128, 64, 3, 2, 1, 1) -- 56
    addConvElement(fusionToRGB, 64, 64, 3, 1, 1) -- 56
    
    addUpConvElement(fusionToRGB, 64, 32, 3, 2, 1, 1) -- 112
    
    fusionToRGB:add(nn.SpatialConvolution(32, 3, 3, 3, 1, 1, 1, 1))
    nameLastModParams(fusionToRGB)
    if opt.TVWeight > 0 then
        print('adding RGB TV loss')
        local tvModule = nn.TVLoss(opt.TVWeight, opt.batchSize):float()
        tvModule:cuda()
        fusionToRGB:add(tvModule)
    end
    
    return fusionToRGB
end

local function createThumbnailUpsamplerNet(opt, subnets)
    -- Input nodes
    local grayscaleImage = nn.Identity()():annotate({name = 'grayscaleImage'})
    local thumbnail = nn.Identity()():annotate({name = 'thumbnail'})
    local targetRGB = nn.Identity()():annotate({name = 'targetRGB'})
    local targetContent = nn.Identity()():annotate({name = 'targetContent'})
    
    -- Intermediates
    local grayEncoderOutput = subnets.grayEncoder(grayscaleImage):annotate({name = 'grayEncoderOutput'})
    local thumbnailToFusionOutput = subnets.thumbnailToFusion(thumbnail):annotate({name = 'thumbnailToFusionOutput'})
    local fusionOutput = nn.JoinTable(1, 3)({grayEncoderOutput, thumbnailToFusionOutput}):annotate({name = 'fusionOutput'})
    local RGBOutput = subnets.fusionToRGB(fusionOutput):annotate({name = 'RGBOutput'})
    
    -- Losses
    
    print('adding pixel RGB loss')
    local pixelRGBLoss = nn.MSECriterion()({RGBOutput, targetRGB}):annotate({name = 'pixelRGBLoss'})
    
    print('adding content loss')
    local perceptualContent = subnets.vggNet(RGBOutput):annotate({name = 'perceptualContent'})
    local contentLoss = nn.MSECriterion()({perceptualContent, targetContent}):annotate({name = 'contentLoss'})

    local pixelRGBLossMul = nn.MulConstant(opt.pixelRGBWeight, true)(pixelRGBLoss)
    local contentLossMul = nn.MulConstant(opt.contentWeight, true)(contentLoss)

    -- Full training network including all loss functions
    local thumbnailUpsamplerNet = nn.gModule({grayscaleImage, thumbnail, targetRGB, targetContent}, {pixelRGBLossMul, contentLossMul})

    cudnn.convert(thumbnailUpsamplerNet, cudnn)
    thumbnailUpsamplerNet = thumbnailUpsamplerNet:cuda()
    graph.dot(thumbnailUpsamplerNet.fg, 'thumbnailUpsamplerNet', 'thumbnailUpsamplerNet')
    return thumbnailUpsamplerNet
end

local function createThumbnailUpsamplerEvalNet(opt, subnets)
    -- Input nodes
    local grayscaleImage = nn.Identity()():annotate({name = 'grayscaleImage'})
    local thumbnail = nn.Identity()():annotate({name = 'thumbnail'})
    
    -- Intermediates
    local grayEncoderOutput = subnets.grayEncoder(grayscaleImage):annotate({name = 'grayEncoderOutput'})
    local thumbnailToFusionOutput = subnets.thumbnailToFusion(thumbnail):annotate({name = 'thumbnailToFusionOutput'})
    local fusionOutput = nn.JoinTable(1, 3)({grayEncoderOutput, thumbnailToFusionOutput}):annotate({name = 'fusionOutput'})
    local RGBOutput = subnets.fusionToRGB(fusionOutput):annotate({name = 'RGBOutput'})
    
    -- Full training network including all loss functions
    local thumbnailUpsamplerEvalNet = nn.gModule({grayscaleImage, thumbnail}, {RGBOutput})

    cudnn.convert(thumbnailUpsamplerEvalNet, cudnn)
    thumbnailUpsamplerEvalNet = thumbnailUpsamplerEvalNet:cuda()
    graph.dot(thumbnailUpsamplerEvalNet.fg, 'thumbnailUpsamplerEvalNet', 'thumbnailUpsamplerEvalNet')
    return thumbnailUpsamplerEvalNet
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
    
    addConvElement(guesserEncoder, 256, 50, 3, 2, 1) -- 14
    
    guesserEncoder:add(nn.Reshape(9800, true))
    
    --guesserEncoder:add(nn.Dropout(0.5))
    addLinearElement(guesserEncoder, 9800, opt.thumbnailFeatures)
    
    guesserEncoder:add(nn.Linear(opt.thumbnailFeatures, opt.thumbnailFeatures))
    nameLastModParams(guesserEncoder)
    
    guesserEncoder:add(nn.Reshape(3, opt.thumbnailSize, opt.thumbnailSize, true))
    
    return guesserEncoder
end

local function craeteDiscriminator(opt)
    local discriminator = nn.Sequential()
    discriminator.paramName = 'discriminator'

    addLinearElementNoBN(discriminator, opt.thumbnailFeatures, opt.thumbnailFeatures)
    addLinearElementNoBN(discriminator, opt.thumbnailFeatures, 1024)
    addLinearElementNoBN(discriminator, 1024, 512)
    addLinearElementNoBN(discriminator, 512, 256)
    
    discriminator:add(nn.Linear(256, 2))
    nameLastModParams(discriminator)
    
    return discriminator
end

local function createDiscriminatorNet(opt, subnets)
    -- Input nodes
    local thumbnails = nn.Identity()():annotate({name = 'thumbnails'})
    local targetCategories = nn.Identity()():annotate({name = 'targetCategories'})
    
    -- Intermediates
    local classProbabilitiesPreLog = subnets.discriminator(thumbnails):annotate({name = 'classProbabilitiesPreLog'})
    
    -- Losses
    print('adding class loss')
    local classProbabilities = cudnn.LogSoftMax()(classProbabilitiesPreLog):annotate({name = 'classProbabilities'})
    local classLoss = nn.ClassNLLCriterion()({classProbabilities, targetCategories}):annotate{name = 'classLoss'}
    
    -- Full training network including all loss functions
    local discriminatorNet = nn.gModule({thumbnails, targetCategories}, {classLoss})

    cudnn.convert(discriminatorNet, cudnn)
    discriminatorNet = discriminatorNet:cuda()
    graph.dot(discriminatorNet.fg, 'discriminatorNet', 'discriminatorNet')
    return discriminatorNet, classProbabilities
end

local function createThumbnailGuesserNet(opt, subnets)
    -- Input nodes
    local grayscaleImage = nn.Identity()():annotate({name = 'grayscaleImage'})
    local targetThumbnails = nn.Identity()():annotate({name = 'targetThumbnails'})
    local targetCategories = nn.Identity()():annotate({name = 'targetCategories'})
    
    -- Intermediates
    local thumbnailOutput = subnets.guesserEncoder(grayscaleImage):annotate({name = 'thumbnailOutput'})
    local classProbabilitiesPreLog = subnets.discriminator(thumbnailOutput):annotate({name = 'classProbabilitiesPreLog'})
    
    -- Losses
    print('adding guide loss')
    local guideLoss = nn.MSECriterion()({thumbnailOutput, targetThumbnails}):annotate({name = 'guideLoss'})
    
    print('adding advesary loss')
    local classProbabilities = cudnn.LogSoftMax()(classProbabilitiesPreLog):annotate({name = 'classProbabilities'})
    local advesaryLoss = nn.ClassNLLCriterion()({classProbabilities, targetCategories}):annotate{name = 'classLoss'}
    
    local guideLossMul = nn.MulConstant(opt.guideWeight, true)(guideLoss)
    local advesaryLossMul = nn.MulConstant(opt.advesaryWeight, true)(advesaryLoss)
    
    -- Full training network including all loss functions
    local thumbnailGuesserNet = nn.gModule({grayscaleImage, targetThumbnails, targetCategories}, {guideLossMul, advesaryLossMul})

    cudnn.convert(thumbnailGuesserNet, cudnn)
    thumbnailGuesserNet = thumbnailGuesserNet:cuda()
    graph.dot(thumbnailGuesserNet.fg, 'thumbnailGuesserNet', 'thumbnailGuesserNet')
    return thumbnailGuesserNet, thumbnailOutput, classProbabilities
end

local function createFinalColorizerNet(opt, subnets)
    -- Input nodes
    local grayscaleImage = nn.Identity()():annotate({name = 'grayscaleImage'})
    
    -- Intermediates
    local thumbnailOutput = subnets.guesserEncoder(grayscaleImage):annotate({name = 'thumbnailOutput'})
    local thumbnailToFusionOutput = subnets.thumbnailToFusion(thumbnailOutput):annotate({name = 'thumbnailToFusionOutput'})
    local grayEncoderOutput = subnets.grayEncoder(grayscaleImage):annotate({name = 'grayEncoderOutput'})
    local fusionOutput = nn.JoinTable(1, 3)({grayEncoderOutput, thumbnailToFusionOutput}):annotate({name = 'fusionOutput'})
    local RGBOutput = subnets.fusionToRGB(fusionOutput):annotate({name = 'RGBOutput'})
    
    -- Full training network including all loss functions
    local finalColorizerNet = nn.gModule({grayscaleImage}, {RGBOutput})

    cudnn.convert(finalColorizerNet, cudnn)
    finalColorizerNet = finalColorizerNet:cuda()
    graph.dot(finalColorizerNet.fg, 'finalColorizerNet', 'finalColorizerNet')
    return finalColorizerNet
end

local function createModel(opt)
    print('Creating model')

    -- Return table
    local r = {}

    -- Create individual sub-networks
    local subnets = {
        grayEncoder = createGrayEncoder(opt),
        thumbnailToFusion = createThumbnailToFusion(opt),
        fusionToRGB = createFusionToRGB(opt),
        discriminator = craeteDiscriminator(opt),
        guesserEncoder = createGuesserEncoder(opt),
        vggNet = createVGG(opt)
    }
    r.grayEncoder = subnets.grayEncoder
    r.thumbnailToFusion = subnets.thumbnailToFusion
    r.fusionToRGB = subnets.fusionToRGB
    r.guesserEncoder = subnets.guesserEncoder
    r.vggNet = subnets.vggNet

    -- Create composite nets
    r.thumbnailUpsamplerNet = createThumbnailUpsamplerNet(opt, subnets)
    r.thumbnailUpsamplerEvalNet = createThumbnailUpsamplerEvalNet(opt, subnets)
    r.discriminatorNet, r.discriminatorProbabilities = createDiscriminatorNet(opt, subnets)
    
    r.colorGuesserNet, r.predictedThumbnails, r.guesserClassProbabilities = createThumbnailGuesserNet(opt, subnets)
    r.finalColorizerNet = createFinalColorizerNet(opt, subnets)
    
    --[[local pretrainedColorGuide = torch.load('pretrainedModels/colorGuide' .. opt.colorGuideSize .. '.t7')
    pretrainedColorGuide:clearState()
    transferParams(pretrainedColorGuide, r.colorGuideNet)
    transferParams(pretrainedColorGuide, r.colorGuidePredictionNet)
    transferParams(pretrainedColorGuide, r.finalColorizerNet)]]
        
    return r
end

return {
    createModel = createModel,
}
