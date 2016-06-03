
require 'lfs'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
paths.dofile('util.lua')
paths.dofile('torchUtil.lua')
paths.dofile('nnModules.lua')

function transformImageDirectory(modelFilename, dirIn, dirOut, cropX, cropY, cropW, cropH, resizeW, resizeH)
    local reflectionPadding = 50
    
    lfs.mkdir(dirOut)
    
    print('loading model from ' .. modelFilename)
    transformNetwork = torch.load(modelFilename)
    transformNetwork = transformNetwork:cuda()
    transformNetwork:evaluate()
    
    local imageFilenames = getFileListRecursive(dirIn)
    for _,filename in ipairs(imageFilenames) do
        local img = image.load(filename)
        img = image.crop(img, cropX, cropY, cropX + cropW, cropY + cropH)
        img = image.scale(img, resizeW, resizeH)
        imgSource = img:clone()
        
        img:add(-0.5)
        img = reflectionPadImage(img, reflectionPadding)
    
        local batchInput = torch.CudaTensor(1, 3, img:size()[2], img:size()[3])
        batchInput[1] = img:clone()
    
        img = img:cuda()
        --print('img size: ' .. getSize(img))
        --print(transformNetwork)
        imgStyled = transformNetwork:forward(batchInput)[1]
        imgStyled = caffeDeprocess(imgStyled)
        
        local imgFinal = torch.FloatTensor(3, imgSource:size()[2], imgSource:size()[3] * 2)
        local imgLeft = imgFinal:narrow(3, 1, resizeW)
        local imgRight = imgFinal:narrow(3, 1 + resizeW, resizeW)
        imgLeft:copy(imgStyled)
        imgRight:copy(imgSource)
        
        local outFilename = string.gsub(filename, dirIn, dirOut)
        print('saving ' .. outFilename)
        image.save(outFilename, imgFinal)
    end
end
