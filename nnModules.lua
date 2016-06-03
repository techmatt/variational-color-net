
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

-- from neural_style.lua

-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, target, normalize)
    parent.__init(self)
    self.strength = strength
    self.target = target
    self.normalize = normalize or false
    self.loss = 0
    self.crit = nn.MSECriterion()
end

function ContentLoss:updateOutput(input)
  if self.target and input:nElement() == self.target:nElement() then
    self.loss = self.crit:forward(input, self.target) * self.strength
  else
    print('WARNING: Skipping content loss')
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    self.gradInput = self.crit:backward(input, self.target)
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength, batchSize)
    parent.__init(self)
    self.strength = strength
    self.x_diff = {}
    self.y_diff = {}
    for b = 1, batchSize do
        self.x_diff[b] = torch.Tensor()
        self.y_diff[b] = torch.Tensor()
    end
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input):zero()
    local B, C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
    
    for b = 1, B do
        self.x_diff[b]:resize(3, H - 1, W - 1)
        self.y_diff[b]:resize(3, H - 1, W - 1)
        self.x_diff[b]:copy(input[b][{{}, {1, -2}, {1, -2}}])
        self.x_diff[b]:add(-1, input[b][{{}, {1, -2}, {2, -1}}])
        self.y_diff[b]:copy(input[b][{{}, {1, -2}, {1, -2}}])
        self.y_diff[b]:add(-1, input[b][{{}, {2, -1}, {1, -2}}])
        self.gradInput[b][{{}, {1, -2}, {1, -2}}]:add(self.x_diff[b]):add(self.y_diff[b])
        self.gradInput[b][{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff[b])
        self.gradInput[b][{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff[b])
    end
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
    return self.gradInput
end

-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function caffePreprocess(img)
    local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):mul(256.0)
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img:add(-1, mean_pixel)
    return img
end

-- Undo the above preprocessing.
function caffeDeprocess(img)
    local mean_pixel = torch.CudaTensor({103.939, 116.779, 123.68})
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img = img + mean_pixel
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):div(256.0)
    return img
end