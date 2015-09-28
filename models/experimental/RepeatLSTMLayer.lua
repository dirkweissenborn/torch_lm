require("models/BaseEncoderLayer")
require("models/BPDropout")
local RepeatLSTMLayer = torch.class('RepeatLSTMLayer', 'BaseEncoderLayer')

function RepeatLSTMLayer:__init(params)
  params.layer_type = params.layer_type or 'RepeatLSTMLayer'
  self.depth = params.depth or 1
  self.repeats = params.repeats
  BaseEncoderLayer.__init(self, params, 2*self.depth, 2*self.depth)
end

function RepeatLSTMLayer:create_encoder()
  local inputs = {}
  local next_s = {}
  table.insert(inputs, nn.Identity()()) -- x
  local lins = {}

  for L = 1,self.depth do
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()
    table.insert(inputs, prev_c)
    table.insert(inputs, prev_h)
    local x
    if L == 1 then x = nn.BPDropout(self.dropout)(inputs[1])
    else x = next_s[2*L-2] end

    local in_size = self.in_capacity
    if L > 1 then in_size = self.capacity end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(in_size, 4 * self.capacity)(x)
    lins[L] ={}
    for j=1, self.repeats do
      local h2h = nn.Linear(self.capacity, 4 * self.capacity)
      lins[L][j] = h2h
      local all_input_sums = nn.CAddTable()({i2h, h2h(prev_h)})

      local reshaped = nn.Reshape(4, self.capacity)(all_input_sums)
      local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
      -- decode the gates
      local in_gate = nn.Sigmoid()(n1)
      local forget_gate = nn.Sigmoid()(n2)
      local out_gate = nn.Sigmoid()(n3)
      -- decode the write inputs
      local in_transform = nn.Tanh()(n4)
      -- perform the LSTM update
      prev_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c }),
        nn.CMulTable()({in_gate,     in_transform})
      })
      -- gated cells form the output
      prev_h = nn.CMulTable()({out_gate, nn.Tanh()(prev_c)})
      table.insert(next_s, prev_c)
      table.insert(next_s, prev_h)
    end
  end

  local m          = nn.gModule(inputs, next_s)
  m = transfer_data(m)
  for L = 1,self.depth do
    for j=2, self.repeats do lins[L][j]:share(lins[L][1],'weight','bias', 'gradWeight', 'gradBias') end
  end
  return m
end

function RepeatLSTMLayer:reset()
  util.zero_table(self.start_s)
  self:reset_s()
end

function RepeatLSTMLayer:fp(prev_l, next_l, length, state)
  util.replace_table(self.s[0], self.start_s)
  local last_s = {}
  for i = 1, length/self.repeats do
    local inp = prev_l.out_s[(i-1)*self.repeats+1]
    local lstm = self:encoder(i)
    local tmp = lstm:forward({inp, unpack(self.s[(i-1)*self.repeats])})

    for j = 1, self.repeats do
      local index = (i-1)*self.repeats+j
      for L = 1,self.depth do
        last_s[2*L-1] = tmp[2*(self.repeats*(L-1)+j)-1]
        last_s[2*L] = tmp[2*(self.repeats*(L-1)+j)]
      end
      util.add_table(self.s[index],last_s)
    end
  end
  util.replace_table(self.start_s, last_s)
  return 0
end

function RepeatLSTMLayer:bp(prev_l, next_l, length, state)
  local last_ds = {}

  for i = length/self.repeats,1,-1 do
    local inp = prev_l.out_s[(i-1)*self.repeats+1]
    local lstm = self:encoder(i)

    for j = 1, self.repeats do
      local index = (i-1)*self.repeats+j
      for L = 1,self.depth do
        last_ds[2*(self.repeats*(L-1)+j)-1] = self.ds[index][2*L-1]
        last_ds[2*(self.repeats*(L-1)+j)] = self.ds[index][2*L]
      end
    end

    local ds = lstm:backward({inp, unpack(self.s[(i-1)*self.repeats])}, last_ds)
    prev_l.out_ds[(i-1)*self.repeats+1]:add(ds[1])
    for L=1,2*self.depth do self.ds[(i-1)*self.repeats][L]:add(ds[L+1]) end
  end
end

function RepeatLSTMLayer:setup(batch_size)
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

function RepeatLSTMLayer:params()
  local t = BaseEncoderLayer.params(self)
  t.depth         = self.depth
  t.type          = self.type
  t.repeats       = self.repeats
  return t
end

function RepeatLSTMLayer:set_params(t)
  BaseEncoderLayer.set_params(self,t)
  self.depth          = t.depth
  self.type           = t.type
  self.repeats        = t.repeats
end

function RepeatLSTMLayer:encoder(i)
  if i*self.repeats > #self.out_s then
    local new_encs
    if self.core_encoder then
      new_encs = util.cloneManyTimes(self.core_encoder,math.ceil(i-#self.encoders))
    end
    for j = 1, i*self.repeats-#self.out_s do
      if self.core_encoder and new_encs[j] then table.insert(self.encoders, new_encs[j]) end
      local s = {}
      local ds = {}
      for d = 1, self.s_size do
        s[d] = transfer_data(torch.zeros(self.batch_size,self.capacity))
        ds[d] = transfer_data(torch.zeros(self.batch_size,self.capacity))
      end
      table.insert(self.s, s)
      table.insert(self.ds, ds)
      table.insert(self.out_s, s[self.out_index])
      table.insert(self.out_ds, ds[self.out_index])
    end
  end
  return self.encoders[i]
end

--- untied weights

local UntiedRepeatLSTMLayer = torch.class('UntiedRepeatLSTMLayer', 'BaseEncoderLayer')

function UntiedRepeatLSTMLayer:__init(params)
  params.layer_type = params.layer_type or 'UntiedRepeatLSTMLayer'
  RepeatLSTMLayer.__init(self, params)
end

function UntiedRepeatLSTMLayer:create_encoder()
  local inputs = {}
  local next_s = {}
  table.insert(inputs, nn.Identity()()) -- x
  local lins = {}

  for L = 1,self.depth do
    local next_c         = nn.Identity()()
    local next_h         = nn.Identity()()
    table.insert(inputs, next_c)
    table.insert(inputs, next_h)
    local x = inputs[2*L-1]
    if L == 1 then x = nn.BPDropout(self.dropout)(x) end

    local in_size = self.in_capacity
    if L > 1 then in_size = self.capacity end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(in_size, 4 * self.capacity)(x)
    lins[L] ={}
    for j=1, self.repeats do
      local h2h = nn.Linear(self.capacity, 4 * self.capacity)
      lins[L][j] = h2h
      local all_input_sums = nn.CAddTable()({i2h, h2h(next_h)})

      local reshaped = nn.Reshape(4, self.capacity)(all_input_sums)
      local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
      -- decode the gates
      local in_gate = nn.Sigmoid()(n1)
      local forget_gate = nn.Sigmoid()(n2)
      local out_gate = nn.Sigmoid()(n3)
      -- decode the write inputs
      local in_transform = nn.Tanh()(n4)
      -- perform the LSTM update
      next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, next_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
      -- gated cells form the output
      next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
      table.insert(next_s, next_c)
      table.insert(next_s, next_h)
    end
  end

  local m          = nn.gModule(inputs, next_s)
  m = transfer_data(m)
  for L = 1,self.depth do
    for j=3, self.repeats do lins[L][j]:share(lins[L][1],'weight','bias', 'gradWeight', 'gradBias') end
  end
  return m
end