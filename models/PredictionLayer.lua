require("models/BaseEncoderLayer")
require("models/BPDropout")
local PredictionLayer = torch.class('PredictionLayer', 'BaseEncoderLayer')

function PredictionLayer:__init(params)
  params.layer_type = params.layer_type or 'PredictionLayer'
  self.repeats = params.repeats or 1
  BaseEncoderLayer.__init(self, params, 1, 1)
end

function PredictionLayer:create_encoder()
  local x = nn.Identity()()
  local y = nn.Identity()()

  local dropped          = nn.BPDropout(self.dropout)(x)
  local h2y              = nn.Linear(self.in_capacity, self.capacity)(dropped)
  local pred             = nn.LogSoftMax()(h2y)
  local err              = nn.ClassNLLCriterion()({pred, y})

  local m                = nn.gModule({x, y}, {pred, err})
  return transfer_data(m)
end

function PredictionLayer:fp(prev_l, next_l, length, state)
  self:encoder(length)
  local loss = 0
  for i = 1, length do
    local inp  = prev_l.out_s[i]
    local y = state.y[(state.pos+ math.floor((i-1) / self.repeats)) % state.y:size(1) + 1]
    local pred = self:encoder(i)
    local tmp_s, tmp = unpack(pred:forward({inp, y}))
    self.out_s[i]:add(tmp_s)
    if self.train or (i % self.repeats) == 0 then loss = loss + tmp[1] end
  end
  return loss
end

function PredictionLayer:bp(prev_l, next_l, length, state)
  for i = length,1,-1 do
    local inp   = prev_l.out_s[i]
    local y     = state.y[(state.pos+ math.floor((i-1) / self.repeats)) % state.y:size(1) + 1]
    local pred  = self:encoder(i)
    local in_ds = pred:backward({inp,y}, {self.out_ds[i],self.pred_err})[1]
    prev_l.out_ds[i]:add(in_ds)
  end
end

function PredictionLayer:setup(batch_size)
  BaseEncoderLayer.setup(self,batch_size)
  self.pred_err = transfer_data(torch.ones(1))
end

function PredictionLayer:params()
  local t = BaseEncoderLayer.params(self)
  t.repeats = self.repeats
  return t
end

function PredictionLayer:set_params(t)
  BaseEncoderLayer.set_params(self,t)
  self.repeats = t.repeats or 1
end