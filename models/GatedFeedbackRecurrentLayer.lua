
require("models/BaseEncoderLayer")
require("pl")
require("models/BPDropout")
local GatedFeedbackRecurrentLayer = torch.class('GatedFeedbackRecurrentLayer', 'BaseEncoderLayer')

function GatedFeedbackRecurrentLayer:__init(params)
  params.layer_type = params.layer_type or 'GatedFeedbackRecurrentLayer'
  self.depth = params.depth or 2
  self.type   = params.type or "gru"
  BaseEncoderLayer.__init(self, params, self.depth, self.depth)
end

function GatedFeedbackRecurrentLayer:create_encoder()
  local x                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = { [0] = x }

  local next_s        = {}
  local split         = { prev_s:split(self.depth) }
  local prev_hs = {}
  for layer_idx = 1, self.depth do table.insert(prev_hs, split[layer_idx]) end

  for layer_idx = 1, self.depth do
    local prev_c         = split[layer_idx]
    local dropped        = nn.BPDropout(self.dropout)(i[layer_idx - 1])
    local next_c
    if self.type =="gru" then next_c = self:feedback_gru(dropped, prev_hs, prev_c)
    else next_c = self:feedback_rnn(dropped, prev_hs) end
    table.insert(next_s, next_c)
    i[layer_idx] = next_c
  end

  local m          = nn.gModule({x, prev_s}, { nn.Identity()(next_s) })
  return transfer_data(m)
end

function GatedFeedbackRecurrentLayer:reset()
  util.zero_table(self.start_s)
  self:reset_s()
end

function GatedFeedbackRecurrentLayer:fp(prev_l, next_l, length, state)
  util.replace_table(self.s[0], self.start_s)
  for i = 1, length do
    local inp = prev_l.out_s[i]
    local lstm = self:encoder(i)
    local tmp = lstm:forward({inp, self.s[i-1]})
    util.add_table(self.s[i],tmp)
  end
  util.replace_table(self.start_s, self.s[length])
  return 0
end

function GatedFeedbackRecurrentLayer:bp(prev_l, next_l, length, state)
  offset = offset or 0
  for i = length,1,-1 do
    local inp = prev_l.out_s[i]
    local lstm = self:encoder(i)
    local d_input, ds = unpack(lstm:backward({inp, self.s[i-1]}, self.ds[i]))
    util.add_table(self.ds[i-1],ds)
    prev_l.out_ds[i]:add(d_input)
  end
end

function GatedFeedbackRecurrentLayer:setup(batch_size)
  BaseEncoderLayer.setup(self, batch_size)
  self.start_s = {}
  self.s[0] = {}
  self.ds[0] = {}
  for d = 1, self.depth do
    table.insert(self.start_s, transfer_data(torch.zeros(batch_size,self.capacity)))
    table.insert(self.ds[0], transfer_data(torch.zeros(batch_size,self.capacity)))
    table.insert(self.s[0], transfer_data(torch.zeros(batch_size,self.capacity)))
  end
  self.out_s[0] = self.s[0][self.out_index]
  self.out_ds[0] =  self.ds[0][self.out_index]
end

function GatedFeedbackRecurrentLayer:params()
  local t = BaseEncoderLayer.params(self)
  t.depth         = self.depth
  return t
end

function GatedFeedbackRecurrentLayer:set_params(t)
  BaseEncoderLayer.set_params(self,t)
  self.depth           = t.depth
end

function GatedFeedbackRecurrentLayer:feedback_rnn(x, prev_hs)
  --global reset
  local inputs = {}
  local join = nn.JoinTable()(prev_hs)
  for i=1,self.depth do
    local sum = nn.CAddTable()({
      nn.Linear(self.in_capacity, 1)(x),
      nn.Linear(self.capacity * self.depth, 1)(join)
    })
    local reset = nn.Replicate(self.capacity,2)(nn.Reshape(1)(nn.Sigmoid()(sum)))
    table.insert(inputs, nn.CMulTable()({
      reset,
      nn.Linear(self.capacity, self.capacity)(prev_hs[i])
    }))
  end

  table.insert(inputs, nn.Linear(self.in_capacity,self.capacity)(x))
  return nn.Tanh()(nn.CAddTable()(inputs))
end

function GatedFeedbackRecurrentLayer:feedback_gru(x, prev_hs, prev_c)
  -- Calculate all n gates in one go
  local gs = nn.Sigmoid()(nn.CAddTable()({
    nn.Linear(self.in_capacity, 2 * self.capacity)(x),
    nn.Linear(self.capacity, 2 * self.capacity)(prev_c)
  }))

  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates = nn.Reshape(2, self.capacity)(gs)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local reset_gate       = nn.SelectTable(1)(sliced_gates)
  local update_gate      = nn.SelectTable(2)(sliced_gates)
  
  -- feedback
  --global reset
  local global_reset = {}
  local join = nn.JoinTable()(prev_hs)
  for i=1,self.depth do
    local sum = nn.CAddTable()({
      nn.Linear(self.in_capacity, 1)(x),
      nn.Linear(self.capacity * self.depth, 1)(join)
    })
    local reset = nn.Replicate(self.capacity,2)(nn.Reshape(1)(nn.Sigmoid()(sum)))
    table.insert(global_reset,reset)
  end

  --input
  local inputs = {}
  for i=1,self.depth do
    table.insert(inputs, nn.CMulTable()({
      global_reset[i],
      nn.Linear(self.capacity, self.capacity)(prev_hs[i])
    }))
  end
  local gated_sum = nn.CMulTable()({reset_gate, nn.CAddTable()(inputs)})
  local next_c    = nn.Tanh()(nn.CAddTable()({gated_sum, nn.Linear(self.in_capacity,self.capacity)(x)}))
  next_c = nn.CAddTable()({
    prev_c,
    nn.CMulTable()({update_gate, nn.CSubTable()({next_c, prev_c})})
  })
  return next_c
end