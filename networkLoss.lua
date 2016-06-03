
--
-- debug coonfig options
--
local printModel = false
describeNets = false
useResidualBlock = false

require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'image'
require 'lfs'

torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

paths.dofile('util.lua')
paths.dofile('torchUtil.lua')

--local testImg = image.load(opt.styleImage)
--local paddedImg = reflectionPadImage(testImg, 100)
--image.save(opt.outDir .. 'padded.png', paddedImg)

--local allImages = getFileListRecursive('/home/mdfisher/ssd2/ImageNet/CLS-LOC/train/')
--local allImages = getFileListRecursive('/home/mdfisher/ssd2/COCO/train2014/')
--writeAllLines(opt.imageList, allImages)

--paths.dofile('movieProcessor.lua')
--transformImageDirectory(opt.outDir .. 'models/transform.t7', 'data/Sintel/framesIn/', opt.outDir .. 'movieOut256/', 267, 62, 360, 360, 256, 256)

paths.dofile('loadModel.lua')
paths.dofile('imageLoader.lua')
paths.dofile('threadPool.lua')

--print(opt)

fullNetwork, transformNetwork, vggTotalNetwork, vggContentNetwork, TVLossModule, contentLossModule, styleLossModules = createModel()
cudnn.convert(fullNetwork, cudnn)
cudnn.convert(vggContentNetwork, cudnn)

-- 3. Convert model to CUDA
print('==> Converting model to CUDA')
fullNetwork = fullNetwork:cuda()
vggContentNetwork = vggContentNetwork:cuda()

if printModel then
    print('=> Model')
    print(fullNetwork)
end

collectgarbage()
print('imagelist: ', opt.imageList)
local imageLoader = makeImageLoader()

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.outDir)
lfs.mkdir(opt.outDir)
lfs.mkdir(opt.outDir .. 'models/')
lfs.mkdir(opt.outDir .. 'samples/')

paths.dofile('train.lua')

epoch = 1

for i=1,opt.epochCount do
   train(imageLoader)
   epoch = epoch + 1
end

