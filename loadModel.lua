
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
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
