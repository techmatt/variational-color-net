
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Image style transfer using network loss')
    cmd:text()
    cmd:text('Options:')
    
    ------------ Network loss options ---------------
    cmd:option('-outDir', 'out/', 'TODO')
    cmd:option('-imageList', 'data/imageListCOCO.txt', 'TODO')
    cmd:option('-batchSize', 4, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-imageSize', 256, 'Smallest side of the resized image')
    cmd:option('-cropSize', 256, 'Height and Width of image crop to be used as input layer')
    
    cmd:option('-contentWeight', 1.0, 'TODO')
    cmd:option('-TVWeight', 1e-4, 'TODO')
    
    cmd:option('-manualSeed', 2, 'Manually set RNG seed')
    cmd:option('-GPU', 2, 'Default preferred GPU')
    
    ------------- Training options --------------------
    cmd:option('-epochCount',         20,    'Number of total epochs to run')
    cmd:option('-epochSize',       1000, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        8, 'number of donkeys to initialize (data loading threads)')
    
    local opt = cmd:parse(arg or {})
    return opt
end

return M
