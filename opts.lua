
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Image style transfer using network loss')
    cmd:text()
    cmd:text('Options:')
    
    ------------ Network  options ---------------
    cmd:option('-outBaseDir', 'out', 'TODO')
    cmd:option('-imageListBase', 'data/places', 'TODO')
    cmd:option('-batchSize', 20, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-superBatches', 1, 'TODO')
    cmd:option('-imageSize', 256, 'Smallest side of the resized image')
    cmd:option('-cropSize', 224, 'Height and Width of image crop to be used as input layer')
    cmd:option('-sceneCategoryCount', 203, 'TODO')
    
    cmd:option('-contentWeight', 0.0005, 'TODO')
    cmd:option('-pixelRGBWeight', 0.01, 'TODO')
    cmd:option('-pixelLABWeight', 2000.0, 'TODO')
    cmd:option('-classWeight', 10.0, 'TODO')
    cmd:option('-TVWeight', 1e-5, 'TODO')
    cmd:option('-KLDWeight', 1.0, 'TODO')
    cmd:option('-useRandomness', true, 'TODO')
    
    cmd:option('-pretrainedTransformModel', 'out/models/transform14.ty', 'TODO')
    
    cmd:option('-manualSeed', 2, 'Manually set RNG seed')
    cmd:option('-GPU', 2, 'Default preferred GPU')
    
    ------------- Training options --------------------
    cmd:option('-epochCount',      100,    'Number of total epochs to run')
    cmd:option('-epochSize',       5000, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-numTestSamples',  5, 'Number of test samples to render periodically during training')
    
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        8, 'number of donkeys to initialize (data loading threads)')
    
    local opt = cmd:parse(arg or {})
    
    opt.halfCropSize = opt.cropSize / 2
    if not opt.useRandomness then opt.KLDWeight = 0.00001 end -- cannot be 0 because of how backprop works

    assert(opt.numTestSamples <= opt.batchSize, 'numTestSamples must not exceed batchSize')
    
    return opt
end

return M
