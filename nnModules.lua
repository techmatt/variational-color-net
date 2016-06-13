
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

-------------------------------------------------------------------------------

-- KL Divergence loss for a Gaussian (input[1] = mean, input[2] = stddev) against
--    another, unit-variance Gaussian
-- Adapted from https://github.com/willwhitney/dc-ign
-- This is actually a module, not a criterion, b/c that plays better with nngraph

local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Module')

function KLDCriterion:__init()
   parent.__init(self)
   self.output = torch.Tensor(1)
   self.sizeAverage = false
end

function KLDCriterion:updateOutput(input)
    -- 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    self.term1 = self.term1 or input[1].new()
    self.term2 = self.term2 or input[2].new()
    self.term3 = self.term3 or input[2].new()

    self.term1:resizeAs(input[1])
    self.term2:resizeAs(input[2])
    self.term3:resizeAs(input[2])

    -- sigma^2
    self.term1:copy(input[2]):exp()

    -- mu^2
    self.term2:copy(input[1]):pow(2)

    -- 1 + log(sigma^2)
    self.term3:fill(1):add(input[2])

    -- 1 + log(sigma^2) - mu^2
    self.term3:add(-1,self.term2)

    -- 1 + log(sigma^2) - mu^2 - sigma^2
    self.term3:add(-1,self.term1)

    if self.sizeAverage then
      self.term3:div(input[1]:nElement())
   end

    self.output[1] = 0.5 * self.term3:sum()

    return self.output
end

function KLDCriterion:updateGradInput(input, gradOutput)
    self.gradInput = {}

    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[1]:resizeAs(input[1])
    self.gradInput[1]:copy(input[1]):mul(-1)


    self.term = self.term or input[2].new()
    self.term:resizeAs(input[2])
    self.term:copy(input[2])

    -- (- sigma^2 + 1) * 0.5
    self.gradInput[2] = self.term:exp():mul(-1):add(1):mul(0.5)

    if self.sizeAverage then
        self.gradInput:div(input[1]:nElement())
    end

    self.gradInput[1]:mul(gradOutput[1])
    self.gradInput[2]:mul(gradOutput[1])

    return self.gradInput
end

