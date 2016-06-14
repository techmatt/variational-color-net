
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
        return '[nil]'
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
    print('dumping graph')
    --lfs.mkdir(dir)
    local out = assert(io.open(filename, "w"))
    local splitter = ","
    for i, module in ipairs(graph:listModules()) do
        local moduleType = torch.type(module)
        print('module ' .. i .. ': ' .. moduleType)
        out:write(i .. splitter .. moduleType .. splitter .. getSize(module.output) .. splitter)
        out:write(getQuartiles(module.output, 10))
        out:write("\n")
        for a,b in pairs(module) do
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


return {
    getSize = getSize,
    describeNet = describeNet,
    dumpNet = dumpNet,
    dumpGraph = dumpGraph,
    caffePreprocess = caffePreprocess,
    caffeDeprocess = caffeDeprocess,
    saveTensor = saveTensor,
    reflectionPadImage = reflectionPadImage
}

