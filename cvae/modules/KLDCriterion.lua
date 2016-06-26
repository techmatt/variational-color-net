-- Adapted from: https://github.com/RuiShu/cvae
-- Treating it as a module instead of a criterion, b/c that makes more sense

local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Module')

-- Table of four inputs: mu1, logv1, mu2, logv2
function KLDCriterion:updateOutput(inputs)
   
   local mu1 = inputs[1]:clone()
   local logv1 = inputs[2]:clone()
   local mu2 = inputs[3]:clone()
   local logv2 = inputs[4]:clone()

   local v1 = torch.exp(logv1)
   local v2 = torch.exp(logv2)
   
   self.output = (torch.add(logv2, -logv1):add(-1):addcdiv(v1, v2):
                     addcdiv((mu2 - mu1):pow(2), v2))

   return self.output:sum() * 0.5
end

function KLDCriterion:updateGradInput(inputs, gradOutput)
   
   local mu1 = inputs[1]:clone()
   local logv1 = inputs[2]:clone()
   local mu2 = inputs[3]:clone()
   local logv2 = inputs[4]:clone()
   
   local v1 = torch.exp(logv1)
   local v2 = torch.exp(logv2)

   local diff12 = mu1:add(-mu2)
   local dmu1 = torch.cdiv(diff12, v2)
   local dmu2 = torch.cdiv(-diff12, v2)
   local div12 = torch.cdiv(v1, v2)
   local dlogv1 = div12:clone():add(-1):div(2)
   -- be careful: use of inplace
   local dlogv2 = div12:mul(-1):add(1):add(-diff12:pow(2):cdiv(v2)):div(2)

   -- return grad w.r.t. input first
   local gradOut = gradOutput[1]
   self.gradInput = {
      dmu2:mul(gradOut),
      dlogv2:mul(gradOut),
      dmu1:mul(gradOut),
      dlogv1:mul(gradOut)
   }
   return self.gradInput
end

