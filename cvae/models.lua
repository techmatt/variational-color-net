require 'KLDCriterion'
require 'GaussianSampler'

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
	grayscale = nn.Reshape(linearSize, true)(grayscale)
	color = nn.Reshape(2*linearSize, true)(grayscale)

	-- Intermediates
	local encMu, encSigma = subnets.encoder({grayscale, color}):split(2)
	local priorMu, priorSigma = subnets.conditionalPrior(grayscale):split(2)
	local code = subnets.sampler({encMu, encSigma})
	local decodedColor = subnets.decoder({grayscale, code})
	decodedColor = nn.Reshape(imgSize, imgSize, 2, true)(decodedColor)

	-- Losses
	
end

local function createTestNet(opt, subnets)
	
end

-- Compile a nn graph into a module and GPU-ify it
local function compile(graph)

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
		encoder = createEncoder(x_size, y_size, z_size, hidden_size)
		conditionalPrior = createConditionalPrior(x_size, z_size, hidden_size)
		sampler = nn.GaussianSampler()
		decoder = createDecoder(x_size, y_size, z_size, hidden_size)
	}

	local M = {}
	M.trainNet = createTrainNet(opt, subnets)
	M.testNet = createTestNet(opt, subnets)
	return M
end



