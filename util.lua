
local M = {}

function M.getFileListRecursive(dir)
    print('Reading files in ' .. dir)
    local result = {}
    for file in lfs.dir(dir) do
        local path = dir .. '/' .. file
        if lfs.attributes(path,"mode") == "file" then
            result[#result + 1] = path
        elseif lfs.attributes(path,"mode") == "directory" and file:sub(1, 1) ~= '.' then
            --print('directory: ' .. file)
            for k, v in ipairs(M.getFileListRecursive(path)) do
                result[#result + 1] = v
            end
        end
    end
    return result
end

function M.fileExists(file)
    local f = io.open(file, "rb")
    if f then f:close() end
    return f ~= nil
end

function M.readAllLines(file)
    if not M.fileExists(file) then 
        print('file not found: ', file)
        return {}
    end
    local lines = {}
    for line in io.lines(file) do 
        lines[#lines + 1] = line
    end
    return lines
end

function M.writeAllLines(file, lines)
    print('writing to ' .. file)
    local f = assert(io.open(file, "w"))
    for i,line in ipairs(lines) do
        f:write(line .. '\n')
    end
end

function M.zeroPad(int, length)
    return string.format( "%0" .. length .. "d", int )
end

function M.split(str, delim)
    -- Eliminate bad cases...
    if string.find(str, delim) == nil then
        return { str }
    end
    local result = {}
    local pat = "(.-)" .. delim .. "()"
    local nb = 0
    local lastPos
    for part, pos in string.gfind(str, pat) do
        nb = nb + 1
        result[nb] = part
        lastPos = pos
    end
    -- Handle the last field
    result[nb + 1] = string.sub(str, lastPos)
    return result
end

function M.listFilesByDir(dir, outFileBase)
    local files = M.getFileListRecursive(dir)
    local L = {}
    for i, file in ipairs(files) do
        local parts = M.split(file, '/')
        local filename = parts[#parts]
        local folder = parts[#parts - 1]
        if folder == 'outdoor' then folder = parts[#parts - 2] end
        --print('entry: ' .. folder .. ' | ' .. filename)
        if L[folder] == nil then L[folder] = {} end
        table.insert(L[folder], file)
    end
    
    local folderList = {}
    for folder, fileList in pairs(L) do
        table.insert(folderList, folder)
        M.writeAllLines(outFileBase .. M.zeroPad(#folderList, 3) .. '.txt', fileList)
    end
    
    M.writeAllLines(outFileBase .. 'Directory.txt', folderList)
end

return M
