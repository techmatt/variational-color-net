-- Adapted from: https://github.com/RuiShu/cvae
-- Treating it as a module instead of a criterion, b/c that makes more sense

local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Module')

function KLDCriterion:__init(sizeAverage)
   parent.__init(self)
   self.sizeAverage = (sizeAverage == nil) and false or sizeAverage
   self.output = torch.Tensor(1)
end

-- Table of four inputs: priorMu, priorLogV, targetMu, targetLogV
-- Compute KL(p1 | p2) where p1 = target and p2 = prior
function KLDCriterion:updateOutput(inputs)
   
   local mu1 = inputs[3]:clone()
   local logv1 = inputs[4]:clone()
   local mu2 = inputs[1]:clone()
   local logv2 = inputs[2]:clone()

   local v1 = torch.exp(logv1)
   local v2 = torch.exp(logv2)
   
   self.output[1] = (torch.add(logv2, -logv1):add(-1):addcdiv(v1, v2):
                     addcdiv((mu2 - mu1):pow(2), v2)):sum() * 0.5

   if self.sizeAverage then
      self.output[1] = self.output[1] / inputs[1]:nElement()
   end

   return self.output
end

-- According to my math, this technically gives 1/2 the gradient (as in, the mu gradients
--    are supposed to be scaled by 2 and the logv gradients by 1, but he's written it
--    so that mu gradients are scaled by 1 and the logv gradients by 1/2)
function KLDCriterion:updateGradInput(inputs, gradOutput)

   local mu1 = inputs[3]:clone()
   local logv1 = inputs[4]:clone()
   local mu2 = inputs[1]:clone()
   local logv2 = inputs[2]:clone()
   
   local v1 = torch.exp(logv1)
   local v2 = torch.exp(logv2)

   local diff12 = mu1:add(-mu2)
   local dmu1 = torch.cdiv(diff12, v2)
   local dmu2 = torch.cdiv(-diff12, v2)
   local div12 = torch.cdiv(v1, v2)
   local dlogv1 = div12:clone():add(-1):div(2)
   -- be careful: use of inplace
   local dlogv2 = div12:mul(-1):add(1):add(-diff12:pow(2):cdiv(v2)):div(2)

   local mulFac = (type(gradOutput) == 'number') and gradOutput or gradOutput[1]
   if self.sizeAverage then
      mulFac = mulFac / inputs[1]:nElement()
   end
   self.gradInput = {
      dmu2:mul(mulFac),
      dlogv2:mul(mulFac),
      dmu1:mul(mulFac),
      dlogv1:mul(mulFac)
   }

   return self.gradInput
end

