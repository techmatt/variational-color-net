
local function getSize(tensor)
    if not tensor or not tensor.size then
        return '[nil]'
    elseif #tensor:size() == 1 then
        return '[' .. tostring(tensor:size()[1]) .. ']'
    elseif #tensor:size() == 2 then
        return '[' .. tostring(tensor:size()[1]) .. ' ' ..
                      tostring(tensor:size()[2]) .. ']'
    elseif #tensor:size() == 3 then
        return '[' .. tostring(tensor:size()[1]) .. ' ' ..
                      tostring(tensor:size()[2]) .. ' ' ..
                      tostring(tensor:size()[3]) .. ']'
    elseif #tensor:size() == 4 then
        return '[' .. tostring(tensor:size()[1]) .. ' ' ..
                      tostring(tensor:size()[2]) .. ' ' ..
                      tostring(tensor:size()[3]) .. ' ' ..
                      tostring(tensor:size()[4]) .. ']'
    else
        return '[unknown vector size]'
    end
end

local function getQuartiles(tensor, count)
    if not tensor or not tensor.size then
        local r = ''
        for q = 1, count do
            r = r .. ' ' .. ','
        end
        return r
    elseif #tensor:size() >= 1 then
        local e = 1
        for d = 1, #tensor:size() do
            e = e * tensor:size()[d]
        end
        local t = tensor:reshape(e)
        local sorted = torch.sort(t:float())
        local r = ''
        for q = 1, count do
            local index = math.max(math.floor(q * e / count), 1)
            r = r .. sorted[index] .. ','
        end
        return r
    end
end

local function describeNet(network, inputs)
    print('dumping network, input size: ' .. getSize(inputs))
    network:forward(inputs)
    --local subnet = nn.Sequential()
    for i, module in ipairs(network:listModules()) do
        local moduleType = torch.type(module)
        --print('module ' .. i .. ': ' .. moduleType)
        if tostring(moduleType) ~= 'nn.Sequential' and
           tostring(moduleType) ~= 'nn.ConcatTable' then
        --    subnet:add(module)
            print('module ' .. i .. ': ' .. getSize(module.output) .. ': ' .. tostring(module))
        end
    end
end

local function dumpNet(network, inputs, dir)
    lfs.mkdir(dir)
    print('dumping network, input size: ' .. getSize(inputs))
    local subnet = nn.Sequential()
    for i, module in ipairs(network:listModules()) do
        local moduleType = torch.type(module)
        --print('module ' .. i .. ': ' .. moduleType)
        if tostring(moduleType) ~= 'nn.Sequential' then
            subnet:add(module)
            local outputs = subnet:forward(inputs)
            print('module ' .. i .. ': ' .. getSize(outputs) .. ': ' .. tostring(module))
            saveTensor(outputs, dir .. i .. '_' .. moduleType .. '.csv')
        end
    end
end

local function dumpGraph(graph, filename)
    local function trainingDesc(s)
        if s then return 'training'
        else return 'evaluate' end
    end
    print('dumping graph')
    --lfs.mkdir(dir)
    local out = assert(io.open(filename, "w"))
    local splitter = ","
    for i, module in ipairs(graph:listModules()) do
        local moduleType = torch.type(module)
        print('module ' .. i .. ': ' .. moduleType)
        --print(module.mode)
        out:write(i .. splitter .. moduleType .. splitter)
        --out:write(trainingDesc(module.train) .. splitter)
        out:write( getSize(module.output) .. splitter .. getQuartiles(module.output, 10) .. splitter)
        out:write( getSize(module.gradInput) .. splitter .. getQuartiles(module.gradInput, 10) .. splitter)
        out:write("\n")
        for a,b in pairs(module) do
            --if a == '_type' then print(a) print(b) end
            --print(a)
            --print(b)
        end
        --[[if tostring(moduleType) ~= 'nn.Sequential' then
            subnet:add(module)
            local outputs = subnet:forward(inputs)
            print('module ' .. i .. ': ' .. getSize(outputs) .. ': ' .. tostring(module))
            saveTensor(outputs, dir .. i .. '_' .. moduleType .. '.csv')
        end]]
    end
    out:close()
end

local function saveTensor2(tensor, filename)
    local out = assert(io.open(filename, "w"))
    out:write(getSize(tensor) .. '\n')
    local maxDim = 32
    local splitter = ","
    for a=1,math.min(maxDim,tensor:size(1)) do
        for b=1,math.min(maxDim,tensor:size(2)) do
            out:write(tensor[a][b])
            if b == tensor:size(2) or b == maxDim then
                out:write("\n")
            else
                out:write(splitter)
            end
        end
    end
    out:close()
end

local function saveTensor3(tensor, filename)
    local out = assert(io.open(filename, "w"))
    out:write(getSize(tensor) .. '\n')
    local maxDim = 32
    local splitter = ","
    for a=1,math.min(maxDim,tensor:size(1)) do
        for b=1,math.min(maxDim,tensor:size(2)) do
            for c=1,math.min(maxDim,tensor:size(3)) do
                out:write(tensor[a][b][c])
                if c == tensor:size(3) or c == maxDim then
                    out:write("\n")
                else
                    out:write(splitter)
                end
            end
        end
    end
    out:close()
end

local function saveTensor4(tensor, filename)
    local out = assert(io.open(filename, "w"))
    out:write(getSize(tensor) .. '\n')
    
    local maxDim = 32
    
    local splitter = ","
    for a=1,math.min(maxDim,tensor:size(1)) do
        for b=1,math.min(maxDim,tensor:size(2)) do
            for c=1,math.min(maxDim,tensor:size(3)) do
                for d=1,math.min(maxDim,tensor:size(4)) do
                    out:write(tensor[a][b][c][d])
                    if d == tensor:size(4) or d == maxDim then
                        out:write("\n")
                    else
                        out:write(splitter)
                    end
                end
            end
        end
    end

    out:close()
end

local function saveTensor(tensor, filename)
    if tensor:nDimension() == 2 then
        saveTensor2(tensor, filename)
    elseif tensor:nDimension() == 3 then
        saveTensor3(tensor, filename)
    elseif tensor:nDimension() == 4 then
        saveTensor4(tensor, filename)
    else
        print('Unknown tensor!')
    end
end

-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
local function caffePreprocess(img)
    local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):mul(256.0)
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img:add(-1, mean_pixel)
    return img
end

-- Undo the above preprocessing.
local function caffeDeprocess(img)
    local mean_pixel = torch.CudaTensor({103.939, 116.779, 123.68})
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img = img + mean_pixel
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):div(256.0)
    return img
end

local function reflectionPadImageVertical(tensor, p)
    local C, W, H = tensor:size()[1], tensor:size()[2], tensor:size()[3]
    local result = tensor.new(C, W, H + p * 2)
    
    local mid = result:narrow(3, p + 1, H)
    mid:copy(tensor)
    
    for v = 1, p do
        local colIn = tensor:narrow(3, v, 1)
        local colOut = result:narrow(3, p + 1 - v, 1)
        colOut:copy(colIn)
    end
    
    for v = 1, p do
        local colIn = tensor:narrow(3, H + 1 - v, 1)
        local colOut = result:narrow(3, H + p + v, 1)
        colOut:copy(colIn)
    end
    
    return result
end

local function reflectionPadImageHorizontal(tensor, p)
    local C, W, H = tensor:size()[1], tensor:size()[2], tensor:size()[3]
    local result = tensor.new(C, W + p * 2, H)
    
    local mid = result:narrow(2, p + 1, W)
    mid:copy(tensor)
    
    for v = 1, p do
        local rowIn = tensor:narrow(2, v, 1)
        local rowOut = result:narrow(2, p + 1 - v, 1)
        rowOut:copy(rowIn)
    end
    
    for v = 1, p do
        local rowIn = tensor:narrow(2, W + 1 - v, 1)
        local rowOut = result:narrow(2, W + p + v, 1)
        rowOut:copy(rowIn)
    end
    
    return result
end

local function reflectionPadImage(tensor, padding)
    local result = reflectionPadImageVertical(tensor, padding)
    result = reflectionPadImageHorizontal(result, padding)
    return result
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
       moduleType == 'nn.CAddTable' or
       moduleType == 'nn.ConcatTable' then
       return false
    end
    if moduleType == 'cudnn.SpatialConvolution' or
       moduleType == 'cudnn.SpatialFullConvolution' or
       moduleType == 'cudnn.SpatialBatchNormalization' or
       moduleType == 'cudnn.BatchNormalization' or
       moduleType == 'nn.Linear' then
       return true
    end
    assert(false, 'unknown module type: ' .. moduleType)
end

local function transferParams(sourceNetwork, targetNetwork)
    print('transterring parameters')
    local sourceNetworkList = {}
    for i, module in ipairs(sourceNetwork:listModules()) do
        if moduelHasParams(module) then
            assert(module.paramName ~= nil, 'unnamed parameter block in source network: module ' .. i .. ' ' .. tostring(torch.type(module)))
            sourceNetworkList[module.paramName] = module
        end
    end
      
    local paramsLoaded = 0
    
    targetNetwork:replace(function(module)
        if moduelHasParams(module) then
            assert(module.paramName ~= nil, 'unnamed parameter block in target network: ' .. tostring(torch.type(module)))
            if sourceNetworkList[module.paramName] == nil then
                print('no parameters found for ' .. module.paramName)
            else
                paramsLoaded = paramsLoaded + 1
                --print('copying paramters for ' .. module.paramName)
                return sourceNetworkList[module.paramName] --:clone()
            end
        end
        return module
    end)
    
    --[[for i, module in ipairs(targetNetwork:listModules()) do
        if moduelHasParams(module) then
            assert(module.paramName ~= nil, 'unnamed parameter block in target network: module ' .. i .. ' ' .. tostring(torch.type(module)))
            if sourceNetworkList[module.paramName] == nil then
                print('no parameters found for ' .. module.paramName)
            else
                paramsLoaded = paramsLoaded + 1
                --print('copying paramters for ' .. module.paramName)
                --module = sourceNetworkList[module.paramName]:clone()
                
                local modsReplaced = 0
                targetNetwork:replace(function(m)
                    print('replace ' .. tostring(torch.type(m)) .. ' ' .. tostring(m.paramName))
                    if m.paramName == module.paramName then
                        modsReplaced = modsReplaced + 1
                        return sourceNetworkList[module.paramName]:clone()
                    else
                        return m
                    end
                end)
                assert(modsReplaced == 1, 'more than one module replaced')
            end
        end
    end]]
    print(paramsLoaded .. ' module parameters transferred')
end

local function nameLastModParams(network)
    assert(network.paramName ~= nil, 'unnamed network')
    local l = network:listModules()
    local lastMod = l[#l]
    lastMod.paramName = network.paramName .. '_' .. #l .. '_' .. torch.type(lastMod)
end



local function toCPUTensor(t)
    return torch.FloatTensor(t:size()):copy(t)
end

local function yuv2lab(iCPU)
    --local iCPU = toCPUTensor(i)
    return image.rgb2lab( image.yuv2rgb(iCPU) )
end

local function lab2yuv(iCPU)
    --local iCPU = toCPUTensor(i)
    return image.rgb2yuv( image.lab2rgb(iCPU) )
end

local function predictionABToRGB(YImageGPU, ABImageGPU)
    local YImage = toCPUTensor(YImageGPU)
    local ABImage = toCPUTensor(ABImageGPU)
    YImage:add(0.5)
    ABImage:mul(100.0)
    local YRepeated = torch.repeatTensor( YImage, 3, 1, 1 )
    YRepeated[2]:zero()
    YRepeated[3]:zero()
    local luminance = yuv2lab(YRepeated)

    local emptyChannel = ABImage[{{1},{},{}}]:float()

    local I = torch.cat(emptyChannel, ABImage, 1)
                        
    local O = image.scale( I, YImage:size()[2], YImage:size()[3] )
    O[1] = luminance[1]
    return image.lab2rgb( O )
end

local function predictionCorrectedRGB(YImageGPU, RGBImageGPU)
    local YImage = toCPUTensor(YImageGPU)
    local RGBImage = toCPUTensor(RGBImageGPU)
    
    YImage:add(0.5)
    local YRepeated = torch.repeatTensor( YImage, 3, 1, 1 )
    YRepeated[2]:zero()
    YRepeated[3]:zero()
    local luminance = yuv2lab(YRepeated)

    local LABImage = image.rgb2lab(RGBImage)
    local LABImage = image.scale( LABImage, YImage:size()[2], YImage:size()[3] )
    
    LABImage[1] = luminance[1]
    return image.lab2rgb( LABImage )
end



return {
    getSize = getSize,
    describeNet = describeNet,
    dumpNet = dumpNet,
    dumpGraph = dumpGraph,
    caffePreprocess = caffePreprocess,
    caffeDeprocess = caffeDeprocess,
    saveTensor = saveTensor,
    reflectionPadImage = reflectionPadImage,
    moduelHasParams = moduelHasParams,
    transferParams = transferParams,
    nameLastModParams = nameLastModParams,
    predictionCorrectedRGB = predictionCorrectedRGB,
    predictionABToRGB = predictionABToRGB
}

