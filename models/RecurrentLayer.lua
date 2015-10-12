require("models/BaseEncoderLayer")
require("models/BPDropout")
local util = require("util")
local RecurrentLayer = torch.class('RecurrentLayer', 'BaseEncoderLayer')

function RecurrentLayer:__init(params)
  params.layer_type = params.layer_type or 'RecurrentLayer'
  self.type         = params.type or "rnn"
  self.depth        = params.depth or 1
  BaseEncoderLayer.__init(self, params, self.depth, self.depth)
end

function RecurrentLayer:create_encoder()
  local inputs,next_s = create_rnn_encoder(self.type, self.in_capacity, self.capacity, self.depth, self.dropout)
  local m             = nn.gModule(inputs, next_s)
  return transfer_data(m)
end

function RecurrentLayer:reset()
  util.zero_table(self.start_s)
  self:reset_s()
end

function RecurrentLayer:fp(prev_l, next_l, length, state)
  util.replace_table(self.s[0], self.start_s)
  self:encoder(length) -- make sure there are enough encoders
  self.last_length = length
  for i = 1, length do
    local inp = prev_l.out_s[i]
    local rnn = self:encoder(i)
    local tmp = rnn:forward({inp, unpack(self.s[i-1])})
    if self.s_size > 1 then util.add_table(self.s[i],tmp)
    else self.s[i][1]:add(tmp) end
  end
  util.replace_table(self.start_s, self.s[length])
  return 0
end

function RecurrentLayer:bp(prev_l, next_l, length, state)
  for i = length,1,-1 do
    local inp = prev_l.out_s[i]
    local rnn = self:encoder(i)
    local ds
    if self.s_size > 1 then ds = rnn:backward({inp, unpack(self.s[i-1])}, self.ds[i])
    else  ds = rnn:backward({inp, unpack(self.s[i-1])}, self.ds[i][1]) end
    prev_l.out_ds[i]:add(ds[1])
    for L=1,self.s_size do self.ds[i-1][L]:add(ds[L+1]) end
  end
end

function RecurrentLayer:setup(batch_size)
  BaseEncoderLayer.setup(self, batch_size)
  self.start_s = {}
  self.s[0] = {}
  self.ds[0] = {}
  for d = 1, self.s_size do
    table.insert(self.start_s, transfer_data(torch.zeros(batch_size,self.capacity)))
    table.insert(self.ds[0], transfer_data(torch.zeros(batch_size,self.capacity)))
    table.insert(self.s[0], transfer_data(torch.zeros(batch_size,self.capacity)))
  end
  self.out_s[0] = self.s[0][self.out_index]
  self.out_ds[0] =  self.ds[0][self.out_index]
end

function RecurrentLayer:params()
  local t = BaseEncoderLayer.params(self)
  t.depth         = self.depth
  t.type          = self.type
  return t
end

function RecurrentLayer:set_params(t)
  BaseEncoderLayer.set_params(self,t)
  self.depth          = t.depth
  self.type           = t.type
end

function gru(in_size, out_size, x, prev_c)
  -- Calculate all n gates in one go
  local gs = nn.Sigmoid()(nn.CAddTable()({
    nn.Linear(in_size, 2 * out_size)(x),
    nn.Linear(out_size, 2 * out_size)(prev_c)
  }))

  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates = nn.Reshape(2, out_size)(gs)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)

  -- Use select gate to fetch each gate and apply nonlinearity
  local reset_gate       = nn.SelectTable(1)(sliced_gates)
  local update_gate      = nn.SelectTable(2)(sliced_gates)

  local next_c           = nn.Tanh()(nn.CAddTable()({nn.Linear(in_size,out_size)(x),
    nn.CMulTable()({reset_gate, nn.Linear(out_size,out_size)(prev_c)})}))

  next_c = nn.CAddTable()({
    prev_c,
    nn.CMulTable()({update_gate, nn.CSubTable()({next_c, prev_c})})
  })
  return next_c
end

function rnn(in_size, out_size, x, prev_c)
  return nn.Tanh()(nn.CAddTable()({
    nn.Linear(in_size, out_size)(x),
    nn.Linear(out_size, out_size)(prev_c)
  }))
end

function create_rnn_encoder(type, in_capacity, capacity, depth, dropout, input)
  dropout = dropout or 0
  local inputs = {}
  local next_s        = {}
  table.insert(inputs, input or nn.Identity()()) -- x
  for L = 1, depth do
    local prev_h = nn.Identity()()
    table.insert(inputs, prev_h)
    local x
    if L == 1 then x = nn.BPDropout(dropout)(inputs[1])
    else x = next_s[L-1] end
    local in_size = in_capacity
    if L > 1 then in_size = capacity end
    local next_h
    if type == "gru" then next_h = gru(in_size, capacity,x, prev_h)
    elseif type == "rnn" then next_h = rnn(in_size, capacity,x, prev_h) end
    table.insert(next_s, next_h)
  end

  return inputs, next_s
end
