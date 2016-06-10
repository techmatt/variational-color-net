
require('nnModules')

local useResidualBlock = true

local function addConvElement(network,iChannels,oChannels,size,stride,padding)
    network:add(nn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    network:add(nn.ReLU(true))
end

local function addUpConvElement(network,iChannels,oChannels,size,stride,padding,extra)
    network:add(nn.SpatialFullConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding,extra,extra))
    --network:add(nn.SpatialUpSamplingNearest(stride))
    --network:add(nn.SpatialConvolution(iChannels,oChannels,size,size,1,1,padding,padding))
    network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    network:add(nn.ReLU(true))
end

local function addResidualBlock(network,iChannels,oChannels,size,stride,padding)
    --addConvElement(network,iChannels,oChannels,size,stride,padding)
    --addConvElement(network,iChannels,oChannels,size,stride,padding)

    local s = nn.Sequential()
        
    s:add(nn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    s:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    s:add(nn.ReLU(true))
    s:add(nn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    s:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    
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

local function createVGGGraph(opt)
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

    --addConvElement(encoder, 1, 32, 9, 1, 4)
    addConvElement(encoder, 1, 32, 7, 1, 2)
    addConvElement(encoder, 32, 64, 3, 2, 1)
    addConvElement(encoder, 64, 128, 3, 2, 1)

    addResidualBlock(encoder, 128, 128, 3, 1, 1)
    addResidualBlock(encoder, 128, 128, 3, 1, 1)
    --addResidualBlock(encoder, 128, 128, 3, 1, 1)
    --addResidualBlock(encoder, 128, 128, 3, 1, 1)
    --addResidualBlock(encoder, 128, 128, 3, 1, 1)
    
    encoder:add(nn.ReLU(true))

    return encoder
end

local function createDecoder(opt)
    local decoder = nn.Sequential()

    addResidualBlock(decoder, 128, 128, 3, 1, 1)
    --addResidualBlock(decoder, 128, 128, 3, 1, 1)
    --addResidualBlock(decoder, 128, 128, 3, 1, 1)

    addUpConvElement(decoder, 128, 64, 3, 2, 1, 1)
    addUpConvElement(decoder, 64, 32, 3, 2, 1, 1)

    decoder:add(nn.SpatialConvolution(32, 3, 3, 3, 1, 1, 1, 1))

    if opt.TVWeight > 0 then
        print('adding TV loss')
        local tvModule = nn.TVLoss(opt.TVWeight, opt.batchSize):float()
        tvModule:cuda()
        decoder:add(tvModule)
    end

    return decoder
end

local function createClassifier(opt)
    local classificationNet = nn.Sequential()
    classificationNet:add(nn.SpatialConvolution(128, 1, 3, 3, 2, 2, 1, 1))
    classificationNet:add(nn.ReLU(true))
    classificationNet:add(nn.Reshape(opt.batchSize, 784, false))
    classificationNet:add(nn.Linear(784, 512))
    classificationNet:add(nn.ReLU(true))
    classificationNet:add(nn.Linear(512, 256))
    classificationNet:add(nn.ReLU(true))
    classificationNet:add(nn.Linear(256, 205))
    return classificationNet
end

local function createModelGraph(opt)
    print('Creating model')

    -- Return table
    local r = {}

    ---------------------------------------------------------------------------

    -- Create individual sub-networks
    local encoder = createEncoder(opt)
    local decoder = createDecoder(opt)
    local classifier = createClassifier(opt) 

    ---------------------------------------------------------------------------

    -- Network that just predicts grayscale -> color images
    r.predictionNet = nn.Sequential()
    r.predictionNet:add(encoder)
    r.predictionNet:add(decoder)
    cudnn.convert(r.predictionNet, cudnn)
    r.predictionNet = r.predictionNet:cuda()

    ---------------------------------------------------------------------------

    -- Graph for training network

    -- Input nodes
    local grayscaleImage = nn.Identity()():annotate{name = 'grayscaleImage'}
    local colorImage = nn.Identity()():annotate{name = 'colorImage'}
    local targetContent = nn.Identity()():annotate{name = 'targetContent'}
    local targetCategories = nn.Identity()():annotate{name = 'targetCategories'}

    -- Intermediates
    local encoderOutput = encoder(grayscaleImage):annotate{name = 'encoderOutput'}
    local decoderOutput = decoder(encoderOutput):annotate{name = 'decoderOutput'}


    -- Losses
    
    print('adding class loss')
    local encoderOutput = encoder(grayscaleImage):annotate{name = 'encoderOutput'}
    local classProbabilitiesPreLog = classifier(encoderOutput):annotate{name = 'classProbabilitiesPreLog'}
    local classProbabilities = cudnn.LogSoftMax()(classProbabilitiesPreLog):annotate{name = 'classProbabilities'}
    --local classLoss = cudnn.ClassNLLCriterion()({r.classProbabilities, targetCategories}):annotate{name = 'classLoss'}
    local classLoss = nn.CrossEntropyCriterion()({classProbabilitiesPreLog, targetCategories}):annotate{name = 'classLoss'}
    
    print('adding pixel loss')
    local pixelLoss = nn.MSECriterion()({decoderOutput, colorImage}):annotate{name = 'pixelLoss'}

    print('adding content loss')
    r.vggNet = createVGGGraph(opt)  -- Needs to be exposed to gradients be zeroed
    local perceptualContent = r.vggNet(decoderOutput):annotate{name = 'perceptualContent'}
    local contentLoss = nn.MSECriterion()({perceptualContent, targetContent}):annotate{name = 'contentLoss'}

    --[[local jointLoss = nn.ParallelCriterion()
    jointLoss:add(classLoss, 100.0)
    jointLoss:add(pixelLoss, 10.0)
    jointLoss:add(contentLoss, 1.0)
    jointLoss = nn.ModuleFromCriterion(jointLoss)()
    r.trainingNet = nn.gModule({grayscaleImage, colorImage, targetContent, targetCategories}, {jointLoss})
    ]]

    --r.trainingNet = nn.gModule({grayscaleImage, colorImage, targetContent, targetCategories}, {classLoss, pixelLoss, contentLoss})
    
    local classLosMul = nn.MulConstant(opt.classWeight, true)(classLoss)
    local pixelLossMul = nn.MulConstant(opt.pixelWeight, true)(pixelLoss)
    local contentLossMul = nn.MulConstant(opt.contentWeight, true)(contentLoss)

    -- Full training network including all loss functions
    r.trainingNet = nn.gModule({grayscaleImage, colorImage, targetContent, targetCategories}, {classLosMul, pixelLossMul, contentLossMul, classProbabilities})
    --r.trainingNet = nn.gModule({grayscaleImage, colorImage, targetContent}, {pixelLossMul, contentLossMul})
    cudnn.convert(r.trainingNet, cudnn)
    r.trainingNet = r.trainingNet:cuda()
    graph.dot(r.trainingNet.fg, 'graphForward', 'graphForward')
    graph.dot(r.trainingNet.bg, 'graphBackward', 'graphBackward')

    ---------------------------------------------------------------------------
    
    return r
end


return {
    createModelGraph = createModelGraph
}
