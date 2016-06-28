require 'cvae/modules/KLDCriterion'
require 'cvae/modules/GaussianSampler'

require 'nnModules'

-- Just do fully-connected on really small images for now

local function createEncoder(x_size, y_size, z_size, hidden_size)
   -- the encoder network
   local encoder = nn.Sequential()
   encoder:add(nn.JoinTable(1,1))
   encoder:add(nn.Linear(x_size + y_size, hidden_size))
   encoder:add(nn.ReLU(true))

   -- construct mu and log variance in parallel
   local mu_logv = nn.ConcatTable()
   mu_logv:add(nn.Linear(hidden_size, z_size))
   mu_logv:add(nn.Linear(hidden_size, z_size))
   encoder:add(mu_logv)
   return encoder
end

local function createConditionalPrior(x_size, z_size, hidden_size)
   -- the prior network
   local prior = nn.Sequential()
   prior:add(nn.Linear(x_size, hidden_size))
   prior:add(nn.ReLU(true))

   -- construct mu and log variance in parallel
   local mu_logv = nn.ConcatTable()
   mu_logv:add(nn.Linear(hidden_size, z_size))
   mu_logv:add(nn.Linear(hidden_size, z_size))
   prior:add(mu_logv)
   return prior
end

local function createDecoder(x_size, y_size, z_size, hidden_size)
   -- the decoder network
   local decoder = nn.Sequential()
   decoder:add(nn.JoinTable(1,1))
   decoder:add(nn.Linear(x_size + z_size, hidden_size))
   decoder:add(nn.ReLU(true))
   decoder:add(nn.Linear(hidden_size, y_size))
   decoder:add(nn.Sigmoid(true))

   return decoder
end

local function createTrainNet(opt, subnets)
	local imgSize = opt.thumbnailSize
	local linearSize = imgSize*imgSize

	-- Inputs
	local grayscale = nn.Identity()()
	local color = nn.Identity()()
	local grayscaleFlat = nn.Reshape(linearSize, true)(grayscale)
	local colorFlat = nn.Reshape(2*linearSize, true)(color)

	-- Intermediates
	local encMu, encSigma = subnets.encoder({grayscaleFlat, colorFlat}):split(2)
	local priorMu, priorSigma = subnets.conditionalPrior(grayscaleFlat):split(2)
	local code = subnets.sampler({encMu, encSigma})
	local decodedColorFlat = subnets.decoder({grayscaleFlat, code})
	local decodedColor = nn.Reshape(imgSize, imgSize, 2, true)(decodedColorFlat)

	-- Losses
	local reconstructionLoss = nn.BCECriterion()({decodedColor, color})
	-- local reconstructionLoss = nn.MSECriterion()({decodedColor, color})
	local kldLoss = nn.KLDCriterion()({priorMu, priorSigma, encMu, encSigma})

	local net = nn.gModule({grayscale, color}, {reconstructionLoss, kldLoss})
	cudnn.convert(net, cudnn)
	return net:cuda()
end

local function createTestNet(opt, subnets)
	local imgSize = opt.thumbnailSize
	local linearSize = imgSize*imgSize

	-- Inputs
	local grayscale = nn.Identity()()
	local grayscaleFlat = nn.Reshape(linearSize, true)(grayscale)

	-- Intermediates
	local priorMu, priorSigma = subnets.conditionalPrior(grayscaleFlat):split(2)
	local code = subnets.sampler({priorMu, priorSigma})
	local decodedColorFlat = subnets.decoder({grayscaleFlat, code})
	local decodedColor = nn.Reshape(imgSize, imgSize, 2, true)(decodedColorFlat)

	local net = nn.gModule({grayscale}, {decodedColor})
	cudnn.convert(net, cudnn)
	return net:cuda()
end

local function createModel(opt)
	local imgw = opt.thumbnailSize
	-- x is grayscale image (L channel)
	local x_size = imgw * imgw
	-- y is chroma image (a, b channels)
	local y_size = x_size * 2
	-- z is the latent code bottleneck size
	local z_size = 10	-- no idea what this should be...
	-- hidden_size is the size of hidden layers in the NNs
	local hidden_size = 500
	local subnets = {
		encoder = createEncoder(x_size, y_size, z_size, hidden_size),
		conditionalPrior = createConditionalPrior(x_size, z_size, hidden_size),
		sampler = nn.GaussianSampler(),
		decoder = createDecoder(x_size, y_size, z_size, hidden_size)
	}

	local M = {}
	M.trainNet = createTrainNet(opt, subnets)
	M.testNet = createTestNet(opt, subnets)
	return M
end

return {
	createModel = createModel
}


