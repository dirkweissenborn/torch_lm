require("models/BaseEncoderLayer")
require("models/GaussianNoise")

local LookupLayer = torch.class('LookupLayer', 'BaseEncoderLayer')

function LookupLayer:__init(params)
  params.layer_type = params.layer_type or 'LookupLayer'
  self.noise_variance = params.noise_variance
  self.flip_prob      = params.flip_prob or 0
  BaseEncoderLayer.__init(self, params, 1, 1)
end

function LookupLayer:create_encoder()
  local x                = nn.Identity()()
  if self.flip_prob > 0 then x = nn.Flip(self.flip_prob)(x) end

  local l                  = nn.LookupTable(self.in_capacity, self.capacity)(x)
  if self.noise_variance and self.noise_variance > 0 then
    local noisy            = nn.GaussianNoise(self.noise_variance)(l)
    local m                = nn.gModule({x}, { noisy })
    return transfer_data(m)
  else
    local m                = nn.gModule({x}, { l })
    return transfer_data(m)
  end
end

function LookupLayer:fp(prev_l, next_l, length, state)
  self:encoder(length)
  for i = 1, length do
    local inp = state.x[(state.pos + i-1) % state.x:size(1) + 1]
    local lookup = self:encoder(i)
    local tmp = lookup:forward(inp)
    self.out_s[i]:add(tmp)
  end
  return 0
end

function LookupLayer:bp(prev_l, next_l, length, state)
  for i = length,1,-1 do
    local inp = state.x[(state.pos+i-1) % state.x:size(1) + 1]
    local lookup = self:encoder(i)
    local in_ds = lookup:backward(inp, self.out_ds[i])
    if input_ds then input_ds[i]:add(in_ds) end
  end
end

function LookupLayer:params()
  local t = BaseEncoderLayer.params(self)
  t.noise_variance = self.noise_variance
  t.flip_prob = self.flip_prob
  return t
end

function LookupLayer:set_params(t)
  BaseEncoderLayer.set_params(self,t)
  self.noise_variance = t.noise_variance
  self.flip_prob = t.flip_prob
end