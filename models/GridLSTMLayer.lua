require("models/BaseEncoderLayer")
require("models/BPDropout")
require("models/LSTMLayer")
local GridLSTMLayer = torch.class('GridLSTMLayer', 'LSTMLayer')

function GridLSTMLayer:__init(params)
  params.layer_type = params.layer_type or 'GridLSTMLayer'
  self.depth = params.depth or 1
  BaseEncoderLayer.__init(self, params, 2*self.depth+1, 2*self.depth+1)
end

function GridLSTMLayer:create_encoder()
  local inputs = {}
  local next_s        = {}
  table.insert(inputs, nn.Identity()()) -- x
  local x = nn.BPDropout(self.dropout)(inputs[1])
  
  local up_c = nn.Tanh()(nn.Linear(self.in_capacity,self.capacity)(x))
  local up_h = nn.Tanh()(nn.Linear(self.in_capacity,self.capacity)(x))

  local in_lins_up = {}
  local rec_lins_up = {}
  local in_lins = {}
  local rec_lins = {}
  for L = 1,self.depth do    
    local prev_c         = nn.Identity()()
    local prev_h         = nn.Identity()()
    table.insert(inputs, prev_c)
    table.insert(inputs, prev_h)

    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(self.capacity, 4 * self.capacity)
    table.insert(in_lins, i2h)
    local h2h = nn.Linear(self.capacity, 4 * self.capacity)
    table.insert(rec_lins, h2h)
    
    local all_input_sums = nn.CAddTable()({i2h(up_h), h2h(prev_h)})

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

    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    
    ------   Upwards ----------------------------------------
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(self.capacity, 4 * self.capacity)
    table.insert(in_lins_up, i2h)
    local h2h = nn.Linear(self.capacity, 4 * self.capacity)
    table.insert(rec_lins_up, h2h)

    local all_input_sums = nn.CAddTable()({i2h(next_h), h2h(up_h)})

    local reshaped = nn.Reshape(4, self.capacity)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    up_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, up_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })
    -- gated cells form the output
    up_h = nn.CMulTable()({out_gate, nn.Tanh()(up_c)})
  end
  table.insert(next_s, up_h)
  local m          = transfer_data(nn.gModule(inputs, next_s))
  for L = 2,self.depth do
    in_lins[L]:share(in_lins[1],"weight","bias","gradWeight","gradBias")
    rec_lins[L]:share(rec_lins[1],"weight","bias","gradWeight","gradBias")
    in_lins_up[L]:share(in_lins_up[1],"weight","bias","gradWeight","gradBias")
    rec_lins_up[L]:share(rec_lins_up[1],"weight","bias","gradWeight","gradBias")
  end
  return m
end

function GridLSTMLayer:reset()
  util.zero_table(self.start_s)
  self:reset_s()
end

function GridLSTMLayer:fp(prev_l, next_l, length, state)
  util.replace_table(self.s[0], self.start_s)
  self:encoder(length) -- make sure there are enough encoders
  local last_s = {}
  for i = 1, length do
    local inp = prev_l.out_s[i]
    local lstm = self:encoder(i)
    for L=1,2*self.depth do last_s[L] = self.s[i-1][L] end
    local tmp = lstm:forward({inp, unpack(last_s)})
    util.add_table(self.s[i],tmp)
  end
  util.replace_table(self.start_s, last_s)
  return 0
end

function GridLSTMLayer:bp(prev_l, next_l, length, state)
  local last_s = {}
  for i = length,1,-1 do
    local inp = prev_l.out_s[i]
    local lstm = self:encoder(i)
    for L=1,2*self.depth do last_s[L] = self.s[i-1][L] end
    local ds = lstm:backward({inp, unpack(last_s)}, self.ds[i])
    prev_l.out_ds[i]:add(ds[1])
    for L=1,2*self.depth do self.ds[i-1][L]:add(ds[L+1]) end
  end
end