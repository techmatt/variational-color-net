
local threadPool = require('threadPool')
local util = require('util')
local torchUtil = require('torchUtil')

local M = {}

function M.filterAllFileLists(opt)
    for category = 1, opt.sceneCategoryCount do
        local inFile = opt.imageListBase .. util.zeroPad(category, 3) .. '.txt'
        local outFile = opt.imageListBase .. util.zeroPad(category, 3) .. '_filtered.txt'
        torchUtil.filterFileList(inFile, outFile)
    end
end

function M.makeImageLoader(opt)
    print('Initializing images from: ' .. opt.imageListBase)
    
    local result = {}
    result.opt = opt
    result.imageLists = {}
    for category = 1, opt.sceneCategoryCount do
        local list = util.readAllLines(opt.imageListBase .. util.zeroPad(category, 3) .. '_filtered.txt')
        table.insert(result.imageLists, list)
        -- print('category ' .. category .. ' has ' .. #list .. ' images')
    end
    result.donkeys = threadPool.makeThreadPool(opt)
    return result
end

local function loadAndResizeImage(path, opt)
    local loadSize = {3, opt.imageSize, opt.imageSize}
    local input = image.load(path, 3, 'float')

    if input:size(2) == loadSize[2] and input:size(3) == loadSize[3] then
        return input
    end
   
   -- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
    if input:size(3) < input:size(2) then
       input = image.scale(input, loadSize[2], loadSize[3] * input:size(2) / input:size(3))
    else
       input = image.scale(input, loadSize[2] * input:size(3) / input:size(2), loadSize[3])
    end
    return input
end

-- function to load the image, jitter it appropriately (random crops etc.)
local function loadAndCropImage(path, opt)
   local sampleSize = {3, opt.cropSize, opt.cropSize}
   collectgarbage()
   local input = loadAndResizeImage(path, opt)
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[3]
   local oH = sampleSize[2]
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   if iH == oH then h1 = 0 end
   if iW == oW then w1 = 0 end
   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(3) == oW)
   assert(out:size(2) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out) end
   return out
end

function M.sampleBatch(imageLoader)
    local opt = imageLoader.opt
    local imageLists = imageLoader.imageLists
    local donkeys = imageLoader.donkeys

    -- pick an index of the datapoint to load next
    local grayscaleInputs = torch.FloatTensor(opt.batchSize, 1, opt.cropSize, opt.cropSize)
    local RGBTargets = torch.FloatTensor(opt.batchSize, 3, opt.halfCropSize, opt.halfCropSize)
    local ABTargets = torch.FloatTensor(opt.batchSize, 2, opt.halfCropSize, opt.halfCropSize)
    local classLabels = torch.IntTensor(opt.batchSize)
    
    for b = 1, opt.batchSize do
        local imageCategory = math.random( #imageLists )
        classLabels[b] = imageCategory
        local list = imageLists[imageCategory]
        local imageFilename = list[ math.random( #list ) ]
        donkeys:addjob(
            function()
                local sourceImg = loadAndCropImage(imageFilename, opt)

                -- Grayscale image
                local grayscale = image.rgb2y(sourceImg)
                -- y is in the range 0 - 1
                
                --[[local imgGray = torch.FloatTensor(1, opt.cropSize, opt.cropSize):zero()
                grayscale:add(0.299, sourceImg:select(1, 1))
                grayscale:add(0.587, sourceImg:select(1, 2))
                grayscale:add(0.114, sourceImg:select(1, 3))]]
                grayscale:add(-0.5)
                
                local downscaleImg = image.scale(sourceImg, opt.halfCropSize, opt.halfCropSize)
                
                local RGBColor = torchUtil.caffePreprocess(downscaleImg:clone())
                local ABColor = image.rgb2lab(downscaleImg)
                ABColor = ABColor[{{2,3},{},{}}]:clone()
                ABColor:mul(1.0 / 100.0)
                
                return grayscale, RGBColor, ABColor
            end,
            function(grayscale, RGBColor, ABColor)
                grayscaleInputs[b] = grayscale
                RGBTargets[b] = RGBColor
                ABTargets[b] = ABColor
            end)
    end
    donkeys:synchronize()
    
    local batch = {}
    batch.grayscaleInputs = grayscaleInputs
    batch.RGBTargets = RGBTargets
    batch.ABTargets = ABTargets
    batch.classLabels = classLabels
    return batch
end

return M
