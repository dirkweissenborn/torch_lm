require 'nn'
require 'data'

local GaussianNoise, Parent = torch.class('nn.GaussianNoise', 'nn.Module')

function GaussianNoise:__init(variance)
  Parent.__init(self)
  self.v = variance
  self.noise = torch.Tensor()
  self.train = true
end

function GaussianNoise:updateOutput(input)
  if self.v > 0 and self.train then
    self.noise:typeAs(input):resizeAs(input)
    self.noise:normal(0,self.v)
    input:add(self.noise)
  end
  self.output = input
  return self.output
end

function GaussianNoise:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  if self.v > 0 and self.train then
    input = input - self.noise
  end
  return self.gradInput
end



-- FlipLayer

local Flip, Parent = torch.class('nn.Flip', 'nn.Module')

function Flip:__init(flip_prob)
  Parent.__init(self)
  self.flip_prob = flip_prob
  self.train = true
end

function Flip:updateOutput(input)
  self.output = torch.Tensor():typeAs(input):resizeAs(input)
  self.output:copy(input)
  if self.flip_prob > 0 and self.train and torch.bernoulli(self.flip_prob) == 1 then
    self.output:uniform(1,data_params.num_chars_extended+1):floor()
  end
  return self.output
end

function Flip:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  return self.gradInput
end
