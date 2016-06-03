require 'torch'
require 'math'
require 'lfs'

function getSize(tensor)
    if not tensor or not tensor.size then
        return '[nil]'
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

function describeNet(network, inputs)
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

function dumpNet(network, inputs, dir)
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

function saveTensor2(tensor, filename)
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

function saveTensor3(tensor, filename)
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

function saveTensor4(tensor, filename)
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

function saveTensor(tensor, filename)
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

function reflectionPadImageVertical(tensor, p)
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

function reflectionPadImageHorizontal(tensor, p)
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

function reflectionPadImage(tensor, padding)
    local result = reflectionPadImageVertical(tensor, padding)
    result = reflectionPadImageHorizontal(result, padding)
    return result
end

