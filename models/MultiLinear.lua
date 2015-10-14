require("nn")

-- basically adaptions of standard modules, however, with better memory handling, because for attention we deal with
-- tensors of changing sizes

local MultiLinear, parent = torch.class('nn.MultiLinear', 'nn.Module')

function MultiLinear:__init(in_size, out_size)
  parent.__init(self)
  self.out_size = out_size
  self.in_size = in_size
  self.weight = torch.Tensor():resize(self.in_size, self.out_size)
  self.gradWeight = torch.Tensor():resize(self.in_size, self.out_size)
end

function MultiLinear:updateOutput(input)
  local batch_size = input:size(1)
  local multi_size = input:size(2)
  self.outBuff  = self.outBuff or torch.Tensor():typeAs(input):resize(batch_size, multi_size, self.out_size)
  if self.outBuff:size(1) < batch_size then --inc buffer
    self.outBuff:resize(batch_size, self.buff:size(2), self.out_size)
  end
  if self.outBuff:size(2) < multi_size then --inc buffer
    self.outBuff:resize(batch_size, math.floor(1.5 * multi_size), self.out_size)
  end
  self.output = self.output or torch.Tensor():typeAs(self.outBuff)
  self.output:set(self.outBuff)
  self.output:resize(batch_size, multi_size, self.out_size)
  local expand = self.weight:view(1, self.in_size, self.out_size):expand(batch_size, self.in_size, self.out_size)
  torch.bmm(self.output, input, expand)
  return self.output
end

function MultiLinear:updateGradInput(input, gradOutput)
  local batch_size = input:size(1)
  local multi_size = input:size(2)

  self.buff  = self.buff or torch.Tensor():typeAs(input):resize(batch_size, multi_size, self.in_size)
  if batch_size > self.buff:size(1) then
    self.buff:resize(batch_size, self.buff:size(2), self.in_size)
  end
  if not self.buff:size(2) or multi_size > self.buff:size(2) then
    self.buff:resize(batch_size,math.floor(1.5 * multi_size), self.in_size)
  end
  self.buff:zero()

  local expand = self.weight:view(1, self.in_size, self.out_size):expand(input:size(1), self.in_size, self.out_size)
  self.gradInput = self.gradInput or torch.Tensor():typeAs(self.buff)
  self.gradInput:set(self.buff)
  self.gradInput:resize(batch_size, multi_size, self.in_size):zero()
  self.gradInput:baddbmm(gradOutput, expand:transpose(2,3))

  return self.gradInput
end

function MultiLinear:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  local grad = self.gradWeight:view(1,self.in_size, self.out_size):expand(input:size(1),self.in_size,self.out_size)
  torch.baddbmm(grad, 1, grad, scale, input:transpose(2,3), gradOutput)
end

------------------------------------------------------------------------------------------
-- memory efficient
local MultiCAddTable, parent = torch.class('nn.MultiCAddTable', 'nn.CAddTable')

function MultiCAddTable:updateOutput(input)
  local size = input[1]:nElement()
  self.outBuff  = self.outBuff or torch.Tensor():typeAs(input[1]):resize(size)
  if self.outBuff:nElement() < size then --make buffer larger
    self.outBuff:resize(math.floor(1.5 * size))
  end
  self.output = self.output or torch.Tensor():typeAs(self.outBuff)
  self.output:set(self.outBuff)
  self.output:resizeAs(input[1])
  self.output:zero()
  for i=1,#input do self.output:add(input[i]) end
  return self.output
end

function MultiCAddTable:updateGradInput(input, gradOutput)
  local size = input[1]:nElement()
  self.buff = self.buff or {}
  for i=1,#input do
    self.buff[i]  = self.buff[i] or torch.Tensor():typeAs(input[i]):resize(size)
    if self.buff[i]:nElement() < size then   --inc buffer
      self.buff[i]:resize(math.floor(1.5 * size))
    end
    self.gradInput[i] = self.gradInput[i] or torch.Tensor():typeAs(self.buff[i])
    self.gradInput[i]:set(self.buff[i])
    self.gradInput[i]:resizeAs(input[i])
    self.gradInput[i]:copy(gradOutput)
  end

  for i=#input+1, #self.gradInput do
    self.gradInput[i] = nil
  end

  return self.gradInput
end


------------------------------------------------------------------------------------------

local ExpandAs, parent = torch.class('nn.ExpandAs', 'nn.Module')

function ExpandAs:__init(dim)
  parent.__init(self)
  self.dim = dim or 1
  self.gradInput = { torch.Tensor() }
  self.gradInput2 = torch.Tensor()
end

function ExpandAs:updateOutput(input)
  local to_expand = input[1]
  local expand_to = input[2]
  local sizes = {}
  for i=1,expand_to:nDimension() do table.insert(sizes, expand_to:size(i)) end
  sizes[self.dim] = 1
  self.output = to_expand:view(unpack(sizes)):expandAs(expand_to)
  return self.output
end

function ExpandAs:updateGradInput(input, gradOutput)
  local to_expand = input[1]
  local expand_to = input[2]
  
  local sizes = {}
  for i=1,expand_to:nDimension() do table.insert(sizes, expand_to:size(i)) end
  sizes[self.dim] = 1
  self.gradInput2:resize(unpack(sizes)):zero()
  self.gradInput[2] = self.gradInput2:expandAs(expand_to)
  
  if to_expand:nElement() ~= self.gradInput[1]:nElement() then
    self.gradInput[1]:resizeAs(to_expand)
    self.gradInput[1]:zero()
  end
  self.gradInput[1] = gradOutput:sum(self.dim):viewAs(self.gradInput[1])
  return self.gradInput
end


-----------------------------------------------------------------------------------------
local MultiMM, parent = torch.class('nn.MultiMM', 'nn.MM')

function MultiMM:updateGradInput(input, gradOutput)
  assert(#input == 2, 'input must be a pair of tensors')
  self.buff = self.buff or { }
  for i=1,2 do
    local size = input[i]:nElement()
    self.buff[i] = self.buff[i] or torch.Tensor():typeAs(input[i]):resize(size)
    if self.buff[i]:nElement() < size then --make buffer larger
      self.buff[i]:resize(math.floor(1.5 * size))
    end
    self.gradInput[i] = self.gradInput[i] or torch.Tensor():typeAs(self.buff[i])
    self.gradInput[i]:set(self.buff[i])
    self.gradInput[i]:resizeAs(input[i])
    self.gradInput[i]:zero()
  end
  return parent.updateGradInput(self, input, gradOutput)
end
