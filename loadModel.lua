
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'loadcaffe'

paths.dofile('nnModules.lua')

function addConvElement(network,iChannels,oChannels,size,stride,padding)
    network:add(nn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    network:add(nn.ReLU(true))
end

function addUpConvElement(network,iChannels,oChannels,size,stride,padding,extra)
    --network:add(nn.SpatialFullConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding,extra,extra))
    network:add(nn.SpatialUpSamplingNearest(stride))
    network:add(nn.SpatialConvolution(iChannels,oChannels,size,size,1,1,padding,padding))
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
    local styleImage = image.load(opt.styleImage, 3)
    styleImage = image.scale(styleImage, opt.cropSize, 'bilinear')
    local styleImageCaffe = caffePreprocess(styleImage):float()
  
    local styleBatch = torch.FloatTensor(opt.batchSize, 3, styleImageCaffe:size()[2], styleImageCaffe:size()[3])
    for i = 1, opt.batchSize do
        styleBatch[i] = styleImageCaffe:clone()
    end
    
    local contentBatch = torch.FloatTensor(opt.batchSize, 3, opt.cropSize, opt.cropSize)
    
    local styleLossModules = {}
    local contentLossModule = {}
    local vggIn = loadcaffe.load('models/VGG_ILSVRC_19_layers_deploy.prototxt',
                                 'models/VGG_ILSVRC_19_layers.caffemodel', 'nn'):float()

    local vggContentOut = nn.Sequential()
    local vggTotalOut = nn.Sequential()

    local vggDepth = 25
    local contentDepth = 9
    
    local contentName = 'relu2_2'
    
    local styleWeights = {}
    styleWeights['relu1_2'] = 0.5
    styleWeights['relu2_2'] = 0.5
    styleWeights['relu3_3'] = 0.25
    styleWeights['relu4_3'] = 2.0
    
    for i = 1, vggDepth do
        local layer = vggIn:get(i)
        local name = layer.name
        --print('layer ' .. i .. ': ' .. name)
        local layerType = torch.type(layer)
        
        vggTotalOut:add(layer)
        --local isPooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')

        if i <= contentDepth then
            vggContentOut:add(layer)
            if name == contentName then
                print("Setting up content layer" .. i .. ": " .. name)
                local contentTarget = vggTotalOut:forward(contentBatch):clone()
                local norm = false
                contentLossModule = nn.ContentLoss(opt.contentWeight, contentTarget, norm):float()
                vggTotalOut:add(contentLossModule)
            end
        end
        
        if styleWeights[name] then
            print("Setting up style layer" .. i .. ": " .. name)
            local gram = GramMatrixSingleton():float()
            local styleTargetFeatures = vggTotalOut:forward(styleBatch):clone()
            local styleTargetGram = gram:forward(styleTargetFeatures[1]):clone()
            styleTargetGram:div(styleTargetFeatures[1]:nElement())
            local norm = false
            local styleLossModule = nn.StyleLoss(opt.styleWeight * styleWeights[name], styleTargetGram, norm, opt.batchSize):float()
            vggTotalOut:add(styleLossModule)
            styleLossModules[#styleLossModules + 1] = styleLossModule
            --saveTensor(styleTargetGram, opt.outDir .. 'styleTargetGram_' .. name .. '.txt')
        end
    end
    
    vggIn = nil
    collectgarbage()
    return vggContentOut, vggTotalOut, contentLossModule, styleLossModules
end

function createModel()
    print('Creating model')
   
    local r = {}
    local transformNetwork = nn.Sequential()
    local fullNetwork = nn.Sequential()
   
    addConvElement(transformNetwork, 3, 32, 9, 1, 0)
    addConvElement(transformNetwork, 32, 64, 3, 2, 0)
    addConvElement(transformNetwork, 64, 128, 3, 2, 0)

    addResidualBlock(transformNetwork, 128, 128, 3, 1, 0)
    addResidualBlock(transformNetwork, 128, 128, 3, 1, 0)
    addResidualBlock(transformNetwork, 128, 128, 3, 1, 0)
    addResidualBlock(transformNetwork, 128, 128, 3, 1, 0)
    addResidualBlock(transformNetwork, 128, 128, 3, 1, 0)

    addUpConvElement(transformNetwork, 128, 64, 3, 2, 0, 0)
    addUpConvElement(transformNetwork, 64, 32, 3, 2, 0, 0)

    transformNetwork:add(nn.SpatialConvolution(32, 3, 3, 3, 1, 1, 0, 0))

    local vggContentNetwork, vggTotalNetwork, contentLossModule, styleLossModules = createVGG()
    
    fullNetwork:add(transformNetwork)
    
    if opt.TVWeight > 0 then
        print('adding TV loss')
        local tvModule = nn.TVLoss(opt.TVWeight, opt.batchSize):float()
        tvModule:cuda()
        fullNetwork:add(tvModule)
    end
    
    fullNetwork:add(vggTotalNetwork)

    return fullNetwork, transformNetwork, vggTotalNetwork, vggContentNetwork, TVLossModule, contentLossModule, styleLossModules
end
