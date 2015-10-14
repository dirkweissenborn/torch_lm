require("models/BaseEncoderLayer")
require("models/BPDropout")
local model_utils = require("model_utils")
local JoinedPredictionLayer = torch.class('JoinedPredictionLayer', 'BaseEncoderLayer')

function JoinedPredictionLayer:__init(params, layers)
  params.layer_type = params.layer_type or 'JoinedPredictionLayer'
  self.layers  = layers
  BaseEncoderLayer.__init(self, params, 1, 1)
  self.core_proj = {}
  for l=1,#layers do
    table.insert(self.core_proj, self:create_projection(layers[l].capacity,self.in_capacity))
  end
  self.pred_err = transfer_data(torch.ones(1))
end

function JoinedPredictionLayer:create_projection(in_cap, out_cap)
  return transfer_data(nn.Linear(in_cap, out_cap))
end

function JoinedPredictionLayer:create_encoder()
  local xs = {}
  for l=1,#self.layers do table.insert(xs, nn.Identity()()) end
  local x = nn.Tanh()(nn.CAddTable()(xs))
  local inputs = tablex.copy(xs)
  local y = nn.Identity()()
  table.insert(inputs,y)
  local dropped          = nn.BPDropout(self.dropout)(x)
  local h2y              = nn.Linear(self.in_capacity, self.capacity)(dropped)
  local pred             = nn.LogSoftMax()(h2y)
  local err              = nn.ClassNLLCriterion()({pred, y})

  local m                = nn.gModule(inputs, {pred, err})
  return transfer_data(m)
end

function JoinedPredictionLayer:fp(prev_l, next_l, length, state)
  self:encoder(length)
  local loss = 0
  for i = 1, length do
    -- setup variables if not existent
    if not self.proj_s[i] then
      self.proj_s[i] = {}
      local skip = 1
      for l=1,#self.layers do
        local layer = self.layers[l]
        skip = skip * (layer.skip or 1)
        if skip > 1 and i % layer.skip ~= 0 and i>1 then
          table.insert(self.proj_s[i], self.proj_s[i-1][l])
        else  
          table.insert(self.proj_s[i], transfer_data(torch.zeros(self.batch_size,self.in_capacity)))
        end
      end
    end
    -- first calc projections
    local skip = 1
    for l=1,#self.layers do
      local layer = self.layers[l]
      skip = skip * (layer.skip or 1)
      if skip == 1 or i % skip == 0 or i==1 then
        self.proj_s[i][l] = self.proj[l]:forward(layer.out_s[math.floor(i/skip)])
      end
    end
    -- then calc prediction
    local y = state.y[(state.pos+i-1) % state.y:size(1) + 1]
    local pred = self:encoder(i)
    local inp = tablex.copy(self.proj_s[i])
    table.insert(inp,y)
    local tmp_s, tmp_loss = unpack(pred:forward(inp))
    self.out_s[i]:add(tmp_s)
    loss = loss + tmp_loss[1]
  end
  return loss
end

function JoinedPredictionLayer:bp(prev_l, next_l, length, state)
  for i = length,1,-1 do
    local y     = state.y[(state.pos+i-1) % state.y:size(1) + 1]
    local pred  = self:encoder(i)
    local inp = tablex.copy(self.proj_s[i])
    table.insert(inp,y)
    local tmp_ds = pred:backward(inp, {self.out_ds[i],self.pred_err})

    local skip = 1
    for l=1,#self.layers do
      local layer = self.layers[l]
      skip = skip * (layer.skip or 1)
      local in_ds = self.proj[l]:backward(layer.out_s[math.floor(i/skip)], tmp_ds[l])
      layer.out_ds[math.floor(i/skip)]:add(in_ds)
    end
  end
end

function JoinedPredictionLayer:setup(batch_size)
  BaseEncoderLayer.setup(self,batch_size)
  self.pred_err = transfer_data(torch.ones(1))
end

function JoinedPredictionLayer:params()
  local t = BaseEncoderLayer.params(self)
  t.core_proj = self.core_proj
  return t
end

function JoinedPredictionLayer:set_params(t)
  BaseEncoderLayer.set_params(self,t)
  self.core_proj = t.core_proj
end

function JoinedPredictionLayer:setup(batch_size)
  BaseEncoderLayer.setup(self,batch_size)
  self.proj_s  = {}
  self.proj_ds = {}
  self.proj = {}
  for l=1,#self.layers do
    table.insert(self.proj, model_utils.clone_many_times(self.core_proj[l],1)[1])
  end
end

function JoinedPredictionLayer:networks(networks)
  networks = networks or {}
  table.insert(networks,self.core_encoder)
  for l=1,#self.layers do
    table.insert(networks, self.core_proj[l])
  end
  return networks
end