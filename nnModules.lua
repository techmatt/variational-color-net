
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