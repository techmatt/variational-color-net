require 'cvae/modules/KLDCriterion'
require 'cvae/modules/GaussianSampler'

local torchUtil = require('torchUtil')
local nameLastModParams = torchUtil.nameLastModParams
local transferParams = torchUtil.transferParams

local useResidualBlock = true
local useLeakyReLU = true

local function makeReLU()
    if useLeakyReLU then
        return nn.LeakyReLU(true)
    else
        cudnn.ReLU(true)
    end
end

local function addConvElement(network,iChannels,oChannels,size,stride,padding)
    network:add(cudnn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    nameLastModParams(network)
    network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    nameLastModParams(network)
    network:add(makeReLU())
end

local function addLinearElement(network,iChannels,oChannels)
    network:add(nn.Linear(iChannels, oChannels))
    nameLastModParams(network)
    network:add(cudnn.BatchNormalization(oChannels, 1e-3))
    nameLastModParams(network)
    network:add(makeReLU())
end

local function addLinearElementNoBN(network,iChannels,oChannels)
    network:add(nn.Linear(iChannels, oChannels))
    nameLastModParams(network)
    network:add(makeReLU())
end

local function addUpConvElement(network,iChannels,oChannels,size,stride,padding,extra)
    network:add(cudnn.SpatialFullConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding,extra,extra))
    nameLastModParams(network)
    network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    nameLastModParams(network)
    network:add(makeReLU())
end

local function addResidualBlock(network,iChannels,oChannels,size,stride,padding)
    local s = nn.Sequential()
    s.paramName = network.paramName .. '_ResBlock' .. #(network:listModules())
    
    s:add(cudnn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    nameLastModParams(s)
    s:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    nameLastModParams(s)
    s:add(makeReLU())
    s:add(cudnn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
    nameLastModParams(s)
    s:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
    nameLastModParams(s)
    
    if useResidualBlock then
        local block = nn.Sequential()
            :add(nn.ConcatTable()
            :add(s)
            :add(nn.Identity()))
            :add(nn.CAddTable(true))
        network:add(block)
    else
        s:add(makeReLU())
        network:add(s)
    end
end

-----------------------------------------------------------------------------------------

local function createEncoder(graySize, colorSize, codeSize)
	local colorLinearSize = colorSize*colorSize*2

	-- Inputs: grayscale and color image
	local grayscale = nn.Identity()()
	local color = nn.Identity()()

	-- Process the grayscale down to something we can linearize
	local grayProcessor = nn.Sequential()
	grayProcessor.paramName = 'encoder_grayProcessor'
	addConvElement(grayProcessor, 1, 50, 3, 2, 1) -- 112
	addConvElement(grayProcessor, 50, 100, 3, 2, 1) -- 56
	addConvElement(grayProcessor, 100, 200, 3, 2, 1) -- 28
	addConvElement(grayProcessor, 200, 400, 3, 2, 1) -- 14
	addConvElement(grayProcessor, 400, 100, 3, 2, 1) -- 7
	grayProcessor:add(nn.Reshape(4900, true))

	-- Once color and grayscale are fused into one big blob,
	--    construct mu and log variance
	local fusedSize = 4900 + colorLinearSize
	local hiddenSize = (fusedSize + codeSize)/2
	local comboEncoder = nn.Sequential()
	comboEncoder.paramName = 'comboEncoder'
	addLinearElementNoBN(comboEncoder, fusedSize, hiddenSize)
	local mu_logv = nn.ConcatTable()
	mu_logv.paramName = 'encoder_mu_logv'
	mu_logv:add(nn.Linear(hiddenSize, codeSize))
	nameLastModParams(mu_logv)
	mu_logv:add(nn.Linear(hiddenSize, codeSize))
	nameLastModParams(mu_logv)
	comboEncoder:add(mu_logv)

	-- Build graph connecting it all up
	local squishedGray = grayProcessor(grayscale)
	local reshapedColor = nn.Reshape(2*colorSize*colorSize, true)(color)
	local joined = nn.JoinTable(1, 1)({squishedGray, reshapedColor})
	local mu, logv = comboEncoder(joined):split(2)
	return nn.gModule({grayscale, color}, {mu, logv})
end

local function createConditionalPrior(colorSize, codeSize)
	-- Just like the encoder, process the gray down
	-- No color this time, so just immediately construct mu and logv
	local prior = nn.Sequential()
	prior.paramName = 'prior_grayProcessor'
	addConvElement(prior, 1, 50, 3, 2, 1) -- 112
	addConvElement(prior, 50, 100, 3, 2, 1) -- 56
	addConvElement(prior, 100, 200, 3, 2, 1) -- 28
	addConvElement(prior, 200, 400, 3, 2, 1) -- 14
	addConvElement(prior, 400, 100, 3, 2, 1) -- 7
	prior:add(nn.Reshape(4900, true))

	local hiddenSize = (4900 + codeSize)/2
	addLinearElementNoBN(prior, 4900, hiddenSize)
	local mu_logv = nn.ConcatTable()
	mu_logv.paramName = 'prior_mu_logv'
	mu_logv:add(nn.Linear(hiddenSize, codeSize))
	nameLastModParams(mu_logv)
	mu_logv:add(nn.Linear(hiddenSize, codeSize))
	nameLastModParams(mu_logv)
	prior:add(mu_logv)
	return prior
end

local function createDecoder(graySize, colorSize, codeSize)
	-- Simple, for now (and a lot like the encoder):
	-- Process the gray down, fuse with the code, fully connect,
	--    then reshape into thumbnail image

	-- Inputs: grayscale, sampled code
	local grayscale = nn.Identity()()
	local codeSample = nn.Identity()()

	-- Process the grayscale down to something we can linearize
	local grayProcessor = nn.Sequential()
	grayProcessor.paramName = 'decoder_grayProcessor'
	addConvElement(grayProcessor, 1, 50, 3, 2, 1) -- 112
	addConvElement(grayProcessor, 50, 100, 3, 2, 1) -- 56
	addConvElement(grayProcessor, 100, 200, 3, 2, 1) -- 28
	addConvElement(grayProcessor, 200, 400, 3, 2, 1) -- 14
	addConvElement(grayProcessor, 400, 100, 3, 2, 1) -- 7
	grayProcessor:add(nn.Reshape(4900, true))

	-- Once code is concatenated, predict output thumnail image
	local colorLinearSize = 2*colorSize*colorSize
	local fusedSize = 4900 + codeSize
	local hiddenSize = (fusedSize + colorLinearSize)/2
	local decoder = nn.Sequential()
	decoder.paramName = 'decoder'
	addLinearElementNoBN(decoder, fusedSize, hiddenSize)
	decoder:add(nn.Linear(hiddenSize, colorLinearSize))
	nameLastModParams(decoder)
	decoder:add(nn.Sigmoid(true))
	decoder:add(nn.Reshape(2, colorSize, colorSize, true))

	-- Build graph connecting it all
	local squishedGray = grayProcessor(grayscale)
	local joined = nn.JoinTable(1, 1)({squishedGray, codeSample})
	local predictedColor = decoder(joined)
	return nn.gModule({grayscale, codeSample}, {predictedColor})
end

local function createTrainNet(opt, subnets)
	-- Inputs
	local grayscale = nn.Identity()()
	local color = nn.Identity()()

	-- Intermediates
	local encMu, encSigma = subnets.encoder({grayscale, color}):split(2)
	local priorMu, priorSigma = subnets.conditionalPrior(grayscale):split(2)
	local codeSample = subnets.sampler({encMu, encSigma})
	local decodedColor = subnets.decoder({grayscale, codeSample})

	-- Losses
	local sizeAverage = true
	local reconstructionLoss = nn.BCECriterion(nil, sizeAverage)({decodedColor, color})
	-- local reconstructionLoss = nn.MSECriterion(sizeAverage)({decodedColor, color})
	local kldLoss = nn.KLDCriterion(sizeAverage)({priorMu, priorSigma, encMu, encSigma})

	local net = nn.gModule({grayscale, color}, {reconstructionLoss, kldLoss})
	cudnn.convert(net, cudnn)
	return net:cuda()
end

local function createTestNet(opt, subnets)
	-- Inputs
	local grayscale = nn.Identity()()

	-- Intermediates
	local priorMu, priorSigma = subnets.conditionalPrior(grayscale):split(2)
	local codeSample = subnets.sampler({priorMu, priorSigma})
	local decodedColor = subnets.decoder({grayscale, codeSample})

	local net = nn.gModule({grayscale}, {decodedColor})
	cudnn.convert(net, cudnn)
	return net:cuda()
end

local function createModel(opt)
	local graySize = opt.cropSize			-- full size gray
	local colorSize = opt.thumbnailSize 	-- thumbnail color
	local codeSize = 10 					-- no idea what's a good value...

	local subnets = {
		encoder = createEncoder(graySize, colorSize, codeSize),
		conditionalPrior = createConditionalPrior(graySize, codeSize),
		sampler = nn.GaussianSampler(),
		decoder = createDecoder(graySize, colorSize, codeSize)
	}

	local M = {}
	M.trainNet = createTrainNet(opt, subnets)
	M.testNet = createTestNet(opt, subnets)
	return M
end

return {
	createModel = createModel
}


