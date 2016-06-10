
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

local function createModelGraph(opt)
    print('Creating model')
   
    local r = {}
    r.grayscaleImage = nn.Identity()():annotate{name = 'grayscaleImage'}
    r.colorImage = nn.Identity()():annotate{name = 'colorImage'}
    r.targetContent = nn.Identity()():annotate{name = 'targetContent'}
    r.targetCategories = nn.Identity()():annotate{name = 'targetCategories'}
    
    r.encoder = nn.Sequential()
    r.decoder = nn.Sequential()
    r.classificationNet = nn.Sequential()
    
    addConvElement(r.encoder, 1, 32, 9, 1, 4)
    addConvElement(r.encoder, 32, 64, 3, 2, 1)
    addConvElement(r.encoder, 64, 128, 3, 2, 1)

    addResidualBlock(r.encoder, 128, 128, 3, 1, 1)
    addResidualBlock(r.encoder, 128, 128, 3, 1, 1)
    addResidualBlock(r.encoder, 128, 128, 3, 1, 1)
    addResidualBlock(r.encoder, 128, 128, 3, 1, 1)
    addResidualBlock(r.encoder, 128, 128, 3, 1, 1)
    
    addResidualBlock(r.decoder, 128, 128, 3, 1, 1)
    addResidualBlock(r.decoder, 128, 128, 3, 1, 1)
    addResidualBlock(r.decoder, 128, 128, 3, 1, 1)

    addUpConvElement(r.decoder, 128, 64, 3, 2, 1, 1)
    addUpConvElement(r.decoder, 64, 32, 3, 2, 1, 1)

    r.decoder:add(nn.SpatialConvolution(32, 3, 3, 3, 1, 1, 1, 1))

    if opt.TVWeight > 0 then
        print('adding TV loss')
        local tvModule = nn.TVLoss(opt.TVWeight, opt.batchSize):float()
        tvModule:cuda()
        r.decoder:add(tvModule)
    end
    
    r.classificationNet:add(nn.SpatialConvolution(128, 1, 3, 3, 2, 2, 1, 1))
    r.classificationNet:add(nn.ReLU(true))
    r.classificationNet:add(nn.Reshape(opt.batchSize, 784, false))
    r.classificationNet:add(nn.Linear(784, 512))
    r.classificationNet:add(nn.ReLU(true))
    r.classificationNet:add(nn.Linear(512, 256))
    r.classificationNet:add(nn.ReLU(true))
    r.classificationNet:add(nn.Linear(256, 205))
    
    print('adding class loss')
    r.encoderOutput = r.encoder(r.grayscaleImage):annotate{name = 'encoderOutput'}
    r.classProbabilitiesPreLog = r.classificationNet(r.encoderOutput):annotate{name = 'classProbabilitiesPreLog'}
    r.classProbabilities = cudnn.LogSoftMax()(r.classProbabilitiesPreLog):annotate{name = 'classProbabilities'}
    --r.classLoss = cudnn.ClassNLLCriterion()({r.classProbabilities, r.targetCategories}):annotate{name = 'classLoss'}
    r.classLoss = nn.CrossEntropyCriterion()({r.classProbabilitiesPreLog, r.targetCategories}):annotate{name = 'classLoss'}
    
    r.decoderOutput = r.decoder(r.encoderOutput):annotate{name = 'decoderOutput'}

    print('adding pixel loss')
    r.pixelLoss = nn.MSECriterion()({r.decoderOutput, r.colorImage}):annotate{name = 'pixelLoss'}

    r.vggNet = createVGGGraph(opt)
    
    r.perceptualContent = r.vggNet(r.decoderOutput):annotate{name = 'perceptualContent'}
    
    print('adding content loss')
    r.contentLoss = nn.MSECriterion()({r.perceptualContent, r.targetContent}):annotate{name = 'contentLoss'}

    --[[r.jointLoss = nn.ParallelCriterion()
    r.jointLoss:add(r.classLoss, 100.0)
    r.jointLoss:add(r.pixelLoss, 10.0)
    r.jointLoss:add(r.contentLoss, 1.0)
    r.jointLoss = nn.ModuleFromCriterion(r.jointLoss)()
    r.graph = nn.gModule({r.grayscaleImage, r.colorImage, r.targetContent, r.targetCategories}, {r.jointLoss})
    ]]
    
    --r.graph = nn.gModule({r.grayscaleImage, r.colorImage, r.targetContent, r.targetCategories}, {r.classLoss, r.pixelLoss, r.contentLoss})
    
    r.classLosMul = nn.MulConstant(opt.classWeight, true)(r.classLoss)
    r.pixelLossMul = nn.MulConstant(opt.pixelWeight, true)(r.pixelLoss)
    r.contentLossMul = nn.MulConstant(opt.contentWeight, true)(r.contentLoss)
    r.graph = nn.gModule({r.grayscaleImage, r.colorImage, r.targetContent, r.targetCategories}, {r.classLosMul, r.pixelLossMul, r.contentLossMul, r.classProbabilities})
    --r.graph = nn.gModule({r.grayscaleImage, r.colorImage, r.targetContent}, {r.pixelLossMul, r.contentLossMul})
    
    cudnn.convert(r.graph, cudnn)
    
    r.graph = r.graph:cuda()

    graph.dot(r.graph.fg, 'graphForward', 'graphForward')
    graph.dot(r.graph.bg, 'graphBackward', 'graphBackward')
    
    return r
end


return {
    createModelGraph = createModelGraph
}
