local BPDropout, Parent = torch.class('nn.BPDropout', 'nn.Module')

function BPDropout:__init(p,v1)
  Parent.__init(self)
  self.p = p or 0.5
  self.train = true
  -- version 2 scales output during training instead of evaluation
  self.v2 = not v1
  if self.p >= 1 or self.p < 0 then
    error('<Dropout> illegal percentage, must be 0 <= p < 1')
  end
  self.noise = torch.Tensor()
end

function BPDropout:updateOutput(input)
  self.output:resizeAs(input):copy(input)
  if self.train then
    self.noise:resizeAs(input)
    self.noise:bernoulli(1-self.p)
    if self.v2 then
      self.noise:div(1-self.p)
    end
    self.output:cmul(self.noise)
  elseif not self.v2 then
    self.output:mul(1-self.p)
  end
  return self.output
end

function BPDropout:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  if self.train then 
    self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
  --else backprop is allowed: Why not? We just don't apply it anymore, same as setting it to 0
    --error('backprop only defined while training')
  end
  return self.gradInput
end

function BPDropout:setp(p)
  self.p = p
end

function BPDropout:__tostring__()
  return string.format('%s(%f)', torch.type(self), self.p)
end
