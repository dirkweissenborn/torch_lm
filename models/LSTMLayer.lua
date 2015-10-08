require("models/BaseEncoderLayer")
require("models/BPDropout")
local util  = require("util")
local LSTMLayer = torch.class('LSTMLayer', 'BaseEncoderLayer')

function LSTMLayer:__init(params)
  params.layer_type = params.layer_type or 'LSTMLayer'
  self.depth = params.depth or 1
  BaseEncoderLayer.__init(self, params, 2*self.depth, 2*self.depth)
end

function LSTMLayer:create_encoder()
  local inputs = {}
  local next_s        = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,self.depth do
    local prev_c         = nn.Identity()() 
    local prev_h         = nn.Identity()()
    table.insert(inputs, prev_c)
    table.insert(inputs, prev_h)
    local x
    if L == 1 then x = nn.BPDropout(self.dropout)(inputs[1]) 
    else x = next_s[2*L-2] end
    
    local in_size = self.in_capacity
    if L > 1 then in_size = self.capacity end
    local next_c, next_h = self:lstm(in_size, x, prev_h, prev_c)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
  end

  local m          = nn.gModule(inputs, next_s)
  return transfer_data(m)
end

function LSTMLayer:reset()
  util.zero_table(self.start_s)
  self:reset_s()
end

function LSTMLayer:fp(prev_l, next_l, length, state)
  util.replace_table(self.s[0], self.start_s)
  self:encoder(length) -- make sure there are enough encoders
  for i = 1, length do
    local inp = prev_l.out_s[i]
    local lstm = self:encoder(i)
    local tmp = lstm:forward({inp, unpack(self.s[i-1])})
    util.add_table(self.s[i],tmp)
  end
  util.replace_table(self.start_s, self.s[length])
  return 0
end

function LSTMLayer:bp(prev_l, next_l, length, state)
  for i = length,1,-1 do
    local inp = prev_l.out_s[i]
    local lstm = self:encoder(i)
    local ds = lstm:backward({inp, unpack(self.s[i-1])}, self.ds[i])
    prev_l.out_ds[i]:add(ds[1])
    for L=1,2*self.depth do self.ds[i-1][L]:add(ds[L+1]) end
  end
end

function LSTMLayer:setup(batch_size)
  BaseEncoderLayer.setup(self, batch_size)
  self.start_s = {}
  self.s[0] = {}
  self.ds[0] = {}
  for d = 1, 2 * self.depth do
    table.insert(self.start_s, transfer_data(torch.zeros(batch_size,self.capacity)))
    table.insert(self.ds[0], transfer_data(torch.zeros(batch_size,self.capacity)))
    table.insert(self.s[0], transfer_data(torch.zeros(batch_size,self.capacity)))
  end
  self.out_s[0] = self.s[0][self.out_index]
  self.out_ds[0] =  self.ds[0][self.out_index]
end

function LSTMLayer:params()
  local t = BaseEncoderLayer.params(self)
  t.depth         = self.depth
  return t
end

function LSTMLayer:set_params(t)
  BaseEncoderLayer.set_params(self,t)
  self.depth          = t.depth
end

function LSTMLayer:lstm(in_size, x, prev_h, prev_c)
  -- evaluate the input sums at once for efficiency
  local i2h = nn.Linear(in_size, 4 * self.capacity)(x)
  local h2h = nn.Linear(self.capacity, 4 * self.capacity)(prev_h)
  local all_input_sums = nn.CAddTable()({i2h, h2h})

  local reshaped = nn.Reshape(4, self.capacity)(all_input_sums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
  -- decode the gates
  local in_gate = nn.Sigmoid()(n1)
  local forget_gate = nn.Sigmoid()(n2)
  local out_gate = nn.Sigmoid()(n3)
  -- decode the write inputs
  local in_transform = nn.Tanh()(n4)
  -- perform the LSTM update
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_transform})
  })
  -- gated cells form the output
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end
