-- Done once at the beginning of the program
paths.dofile('globals.lua')

--note: using require won't emit parsing errors while paths.dofile will.

local opts = require('opts.lua')
local util = require('util.lua')
local models = require('models.lua')
local imageLoader = require('imageLoader.lua')
local train = paths.dofile('train.lua')


torch.setdefaulttensortype('torch.FloatTensor')

local opt = opts.parse(arg)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

local model = models.createModelGraph(opt)

--local allImages = util.getFileListRecursive('/home/dritchie/mscoco/')
--util.writeAllLines(opt.imageList, allImages)

--util.listFilesByDir('/home/mdfisher/ssd2/Places/images256/', 'data/places')

collectgarbage()
local imgLoader = imageLoader.makeImageLoader(opt)

print('Saving everything to: ' .. opt.outDir)
lfs.mkdir(opt.outDir)
lfs.mkdir(opt.outDir .. 'models/')
lfs.mkdir(opt.outDir .. 'samples/')

for i=1,opt.epochCount do
   train(model, imgLoader, opt, i)
end
