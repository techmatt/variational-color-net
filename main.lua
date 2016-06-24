-- Done once at the beginning of the program
paths.dofile('globals.lua')

--note: using require won't emit parsing errors while paths.dofile will.

local opts = require('opts')

local opt = opts.parse(arg)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')
cudnn.benchmark = true

local util = require('util')
--local models = require('models')
--local modelsColorGuide = require('modelsColorGuide')
local modelsThumbnail = require('modelsThumbnail')
local torchUtil = require('torchUtil')
local imageLoader = require('imageLoader')
--local train = require('train')
--local train = require('trainColorGuide')
--local train = require('trainColorGuesser')
local train = require('trainThumbnailUpsampler')

--local allImages = util.getFileListRecursive('/home/dritchie/mscoco/')
--util.writeAllLines(opt.imageList, allImages)

--util.listFilesByDir('/home/mdfisher/ssd2/Places/images256/', 'data/places')
--imageLoader.filterAllFileLists(opt)

--local model = models.createModel(opt)
--local model = modelsColorGuide.createModel(opt)
local model = modelsThumbnail.createModel(opt)

collectgarbage()
local imgLoader = imageLoader.makeImageLoader(opt)

--torchUtil.vibrancyTest(imgLoader.imageLists, 500, 'vibrancyTest/')

-- Create unique directory for outputs (based on timestamp)
opt.outDir = string.format('%s_%u/', opt.outBaseDir, os.time())
print('Saving everything to: ' .. opt.outDir)
lfs.mkdir(opt.outDir)
lfs.mkdir(opt.outDir .. 'models/')
lfs.mkdir(opt.outDir .. 'samples/')
-- Copy over all .lua files
lfs.mkdir(opt.outDir .. 'src/')
for file in lfs.dir('.') do
	if paths.extname(file) == 'lua' then
		os.execute(string.format('cp %s %s/src/%s', file, opt.outDir, file))
	end
end

for i=1,opt.epochCount do
   train(model, imgLoader, opt, i)
end
