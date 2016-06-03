
require 'lfs'

function getFileListRecursive(dir)
    print('Reading files in ' .. dir)
    local result = {}
    for file in lfs.dir(dir) do
        local path = dir .. '/' .. file
        if lfs.attributes(path,"mode") == "file" then
            result[#result + 1] = path
        elseif lfs.attributes(path,"mode") == "directory" and file:sub(1, 1) ~= '.' then
            print('directory: ' .. file)
            for k, v in ipairs(getFileListRecursive(path)) do
                result[#result + 1] = v
            end
        end
    end
    return result
end

function getDirList(dir)

end

function fileExists(file)
    local f = io.open(file, "rb")
    if f then f:close() end
    return f ~= nil
end

function readAllLines(file)
    if not fileExists(file) then 
        print('file not found: ', file)
        return {}
    end
    local lines = {}
    for line in io.lines(file) do 
        lines[#lines + 1] = line
    end
    return lines
end

function writeAllLines(file, lines)
    local f = assert(io.open(file, "w"))
    for i,line in ipairs(lines) do
        f:write(line .. '\n')
    end
end
