
require('nnModules')

local useResidualBlock = true
local useLeakyReLU = true
local colorGuideSize = 512

local function makeReLU()
    if useLeakyReLU then
        return nn.LeakyReLU(true)
    else
        cudnn.ReLU(true)
    end
end

local function moduelHasParams(module)
    local moduleType = tostring(torch.type(module))
    if moduleType == 'nn.gModule' or
       moduleType == 'nn.Sequential' or
       moduleType == 'nn.Identity' or
       moduleType == 'cudnn.ReLU' or
       moduleType == 'cudnn.SpatialMaxPooling' or
       moduleType == 'nn.ReLU' or
       moduleType == 'nn.LeakyReLU' or
       moduleType == 'nn.Reshape' or
       moduleType == 'cudnn.Tanh' or
       moduleType == 'nn.JoinTable' or
       moduleType == 'nn.TVLoss' or
       moduleType == 'nn.ModuleFromCriterion' or
       moduleType == 'nn.MulConstant' or
       moduleType == 'nn.ConcatTable' then
       return false
    end
    if moduleType == 'cudnn.SpatialConvolution' or
       moduleType == 'cudnn.SpatialFullConvolution' or
       moduleType == 'cudnn.SpatialBatchNormalization' or
       moduleType == 'cudnn.BatchNormalization' or
       moduleType == 'nn.Linear' or
       moduleType == 'nn.Sequential' or
       moduleType == 'nn.Sequential' or
       moduleType == 'nn.yy' then
       return true
    end
    assert(false, 'unknown module type: ' .. moduleType)
end

local function transferParams(sourceNetwork, targetNetwork)
    print('transterring parameters')
    local sourceNetworkList = {}
    for i, module in ipairs(sourceNetwork:listModules()) do
        print(module.paramName)
        if moduelHasParams(module) then
            assert(module.paramName ~= nil, 'unnamed parameter block in source network: module ' .. i .. ' ' .. tostring(torch.type(module)))
            sourceNetworkList[module.paramName] = module
        end
    end
    
    for i, module in ipairs(targetNetwork:listModules()) do
        if moduelHasParams(module) then
            assert(module.paramName ~= nil, 'unnamed parameter block in target network: module ' .. i .. ' ' .. tostring(torch.type(module)))
            if sourceNetworkList[module.paramName] == nil then
                print('no parameters found for ' .. module.paramName)
            else
                print('copying paramters for ' .. module.paramName)
                module = sourceNetworkList[module.paramName]:clone()
            end
        end
    end
end

local function nameLastModParams(network)
    assert(network.paramName ~= nil, 'unnamed network')
    local l = network:listModules()
    local lastMod = l[#l]
    lastMod.paramName = network.paramName .. '_' .. #l .. '_' .. torch.type(lastMod)
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
        
    s:add(cudnn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    nameLastModParams(network)
    s:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    nameLastModParams(network)
    s:add(makeReLU())
    s:add(cudnn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    nameLastModParams(network)
    s:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    nameLastModParams(network)
    
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
    
    colorEncoder:add(nn.Linear(1024, colorGuideSize))
    nameLastModParams(colorEncoder)
    colorEncoder:add(nn.Tanh())
    
    return colorEncoder
end

local function createGuideToFusion(opt)
    local guideToFusion = nn.Sequential()
    guideToFusion.paramName = 'guideToFusion'

    addLinearElement(guideToFusion, colorGuideSize, 784)
    
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

local function createModel(opt)
    print('Creating model')

    -- Return table
    local r = {}

    -- Create individual sub-networks
    local subnets = {
        grayEncoder = createGrayEncoder(opt),
        colorEncoder = createColorEncoder(opt),
        guideToFusion = createGuideToFusion(opt),
        decoder = createDecoder(opt),
        vggNet = createVGG(opt)
    }
    r.grayEncoder = subnets.grayEncoder
    r.colorEncoder = subnets.colorEncoder
    r.guideToFusion = subnets.guideToFusion
    r.decoder = subnets.decoder
    r.vggNet = subnets.vggNet  -- Needs to be exposed to gradients be zeroed

    -- Create composite nets
    r.colorGuideNet = createColorGuideNet(opt, subnets)
    r.colorGuidePredictionNet = createColorGuidePredictionNet(opt, subnets)
    
    local pretrainedColorGuide = torch.load('pretrainedModels/transform1.t7')
    
    transferParams(pretrainedColorGuide, r.colorGuideNet)
    
    return r
end


return {
    createModel = createModel
}
