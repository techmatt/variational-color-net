
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'optim'
require 'nngraph'
require 'loadcaffe'

local useResidualBlock = true

paths.dofile('nnModules.lua')

function addConvElement(network,iChannels,oChannels,size,stride,padding)
    network:add(nn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    network:add(nn.ReLU(true))
end

function addUpConvElement(network,iChannels,oChannels,size,stride,padding,extra)
    network:add(nn.SpatialFullConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding,extra,extra))
    --network:add(nn.SpatialUpSamplingNearest(stride))
    --network:add(nn.SpatialConvolution(iChannels,oChannels,size,size,1,1,padding,padding))
    network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    network:add(nn.ReLU(true))
end

function addResidualBlock(network,iChannels,oChannels,size,stride,padding)
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

function createVGG()
    local contentBatch = torch.FloatTensor(opt.batchSize, 3, opt.cropSize, opt.cropSize)
    
    local contentLossModule = {}
    
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
        if name == contentName then
            print("Setting up content layer" .. i .. ": " .. name)
            local contentTarget = vggContentOut:forward(contentBatch):clone()
            local norm = false
            contentLossModule = nn.ContentLoss(opt.contentWeight, contentTarget, norm):float()
            vggContentOut:add(contentLossModule)
        end
    end
    
    vggIn = nil
    collectgarbage()
    return vggContentOut, contentLossModule
end

function createModel()
    print('Creating model')
   
    local r = {}
    local transformNetwork = nn.Sequential()
    local fullNetwork = nn.Sequential()
    
    addConvElement(transformNetwork, 1, 32, 9, 1, 4)
    addConvElement(transformNetwork, 32, 64, 3, 2, 1)
    addConvElement(transformNetwork, 64, 128, 3, 2, 1)

    addResidualBlock(transformNetwork, 128, 128, 3, 1, 1)
    addResidualBlock(transformNetwork, 128, 128, 3, 1, 1)
    
    addResidualBlock(transformNetwork, 128, 128, 3, 1, 1)
    addResidualBlock(transformNetwork, 128, 128, 3, 1, 1)
    addResidualBlock(transformNetwork, 128, 128, 3, 1, 1)
    addResidualBlock(transformNetwork, 128, 128, 3, 1, 1)
    addResidualBlock(transformNetwork, 128, 128, 3, 1, 1)
    
    addResidualBlock(transformNetwork, 128, 128, 3, 1, 1)
    addResidualBlock(transformNetwork, 128, 128, 3, 1, 1)

    addUpConvElement(transformNetwork, 128, 64, 3, 2, 1, 1)
    addUpConvElement(transformNetwork, 64, 32, 3, 2, 1, 1)

    transformNetwork:add(nn.SpatialConvolution(32, 3, 3, 3, 1, 1, 1, 1))

    local vggContentNetwork, contentLossModule = createVGG()
    
    fullNetwork:add(transformNetwork)
    
    if opt.TVWeight > 0 then
        print('adding TV loss')
        local tvModule = nn.TVLoss(opt.TVWeight, opt.batchSize):float()
        tvModule:cuda()
        fullNetwork:add(tvModule)
    end
    
    local pixelLossModule
    if opt.pixelWeight > 0 then
        print('adding pixel loss')
        local contentBatch = torch.FloatTensor(opt.batchSize, 3, opt.cropSize, opt.cropSize)
        local norm = false
        pixelLossModule = nn.ContentLoss(opt.pixelWeight, contentBatch, norm):float()
        pixelLossModule:cuda()
        fullNetwork:add(pixelLossModule)
    end
    
    fullNetwork:add(vggContentNetwork)

    return fullNetwork, transformNetwork, vggContentNetwork, contentLossModule, pixelLossModule
end

function createVGGGraph()
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

function createModelGraph()
    print('Creating model')
   
    local r = {}
    r.grayscaleImage = nn.Identity()():annotate{name = 'grayscaleImage'}
    r.colorImage = nn.Identity()():annotate{name = 'colorImage'}
    r.targetContent = nn.Identity()():annotate{name = 'targetContent'}
    --r.targetCategories = nn.Identity()():annotate{name = 'targetCategories'}
    
    r.transformNet = nn.Sequential()
    
    addConvElement(r.transformNet, 1, 32, 9, 1, 4)
    addConvElement(r.transformNet, 32, 64, 3, 2, 1)
    addConvElement(r.transformNet, 64, 128, 3, 2, 1)

    addResidualBlock(r.transformNet, 128, 128, 3, 1, 1)
    addResidualBlock(r.transformNet, 128, 128, 3, 1, 1)
    addResidualBlock(r.transformNet, 128, 128, 3, 1, 1)
    addResidualBlock(r.transformNet, 128, 128, 3, 1, 1)
    addResidualBlock(r.transformNet, 128, 128, 3, 1, 1)
    addResidualBlock(r.transformNet, 128, 128, 3, 1, 1)
    addResidualBlock(r.transformNet, 128, 128, 3, 1, 1)
    addResidualBlock(r.transformNet, 128, 128, 3, 1, 1)
    addResidualBlock(r.transformNet, 128, 128, 3, 1, 1)

    addUpConvElement(r.transformNet, 128, 64, 3, 2, 1, 1)
    addUpConvElement(r.transformNet, 64, 32, 3, 2, 1, 1)

    r.transformNet:add(nn.SpatialConvolution(32, 3, 3, 3, 1, 1, 1, 1))

    if opt.TVWeight > 0 then
        print('adding TV loss')
        local tvModule = nn.TVLoss(opt.TVWeight, opt.batchSize):float()
        tvModule:cuda()
        r.transformNet:add(tvModule)
    end
    
    r.transformedImage = r.transformNet(r.grayscaleImage):annotate{name = 'transformedImage'}

    print('adding pixel loss')
    r.pixelLoss = nn.MSECriterion()({r.transformedImage, r.colorImage}):annotate{name = 'pixelLoss'}

    r.vggNet = createVGGGraph()
    
    r.transformedContent = r.vggNet(r.transformedImage):annotate{name = 'transformedContent'}
    
    print('adding content loss')
    r.contentLoss = nn.MSECriterion()({r.transformedContent, r.targetContent}):annotate{name = 'contentLoss'}

    r.graph = nn.gModule({r.grayscaleImage, r.colorImage, r.targetContent}, {r.pixelLoss, r.contentLoss})
    
    cudnn.convert(r.graph, cudnn)
    cudnn.convert(r.transformNet, cudnn)
    cudnn.convert(r.vggNet, cudnn)
    
    r.graph = r.graph:cuda()
    r.transformNet = r.transformNet:cuda()
    r.vggNet = r.vggNet:cuda()

    --graph.dot(r.graph.fg, 'graph', 'graph')
    
    return r
end
