require("nn")
local InvTanh = torch.class('nn.InvTanh', 'nn.Module')

function InvTanh:updateOutput(input)
  self.output:resizeAs(input):copy(input)
  self.tmp = self.tmp or input:clone()
  self.output:add(1)
  self.tmp:fill(1):add(-1,input)
  self.output:cdiv(self.tmp):log():mul(0.5)
  return self.output
end

function InvTanh:updateGradInput(input, gradOutput)
  self.tmp = self.tmp or input:clone()
  self.tmp:copy(input):cmul(input):mul(-1):add(1)
  return self.gradInput:resizeAs(input):fill(1):cdiv(self.tmp):cmul(gradOutput)
end