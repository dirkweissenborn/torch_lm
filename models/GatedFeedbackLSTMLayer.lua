require("models/LSTMLayer")


local GatedFeedbackLSTMLayer = torch.class('GatedFeedbackLSTMLayer', 'LSTMLayer')

local RevGatedFeedbackLSTMLayer = torch.class('RevGatedFeedbackLSTMLayer', 'GatedFeedbackLSTMLayer')

function RevGatedFeedbackLSTMLayer:__init(params)
  params.layer_type = params.layer_type or 'RevGatedFeedbackLSTMLayer'
  self.depth = params.depth or 2
  BaseEncoderLayer.__init(self, params, 2*self.depth, 2)
end

function GatedFeedbackLSTMLayer:__init(params)
  params.layer_type = params.layer_type or 'GatedFeedbackLSTMLayer'
  self.depth = params.depth or 2
  LSTMLayer.__init(self)
end

function GatedFeedbackLSTMLayer:create_encoder()
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  for L = 1,self.depth do
    local prev_c         = nn.Identity()()
    local prev_h         = nn.Identity()()
    table.insert(inputs, prev_c)
    table.insert(inputs, prev_h)
  end

  local next_s        = {}
  local prev_hs = {}
  for L = 1, self.depth do table.insert(prev_hs, inputs[2 * L+1]) end

  for L = 1,self.depth do
    local prev_c         = inputs[2*L]
    local prev_h         = inputs[2*L+1]

    local x
    if L == 1 then x = nn.BPDropout(self.dropout)(inputs[1])
    else x = next_s[2*L-2] end

    local in_size = self.in_capacity
    if L > 1 then in_size = self.capacity end
    local next_c, next_h = self:feedback_lstm(in_size, x, prev_h, prev_hs, prev_c)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
  end

  local m          = nn.gModule(inputs, next_s)
  return transfer_data(m)
end

function GatedFeedbackLSTMLayer:in_transform(in_size, x, prev_hs)
  local inputs = {}
  local join = nn.JoinTable(2)(prev_hs)
  for i=1,self.depth do
    local sum = nn.CAddTable()({
      nn.Linear(in_size, 1)(x),
      nn.Linear(self.capacity * self.depth, 1)(join)
    })
    local reset = nn.Replicate(self.capacity,2)(nn.Sigmoid()(sum))
    table.insert(inputs, nn.CMulTable()({
      reset,
      nn.Linear(self.capacity, self.capacity)(prev_hs[i])
    }))
  end

  table.insert(inputs, nn.Linear(in_size,self.capacity)(x))
  return nn.Tanh()(nn.CAddTable()(inputs))
end

function GatedFeedbackLSTMLayer:feedback_lstm(in_size, x, prev_h, prev_hs, prev_c)
  local in_transform = self:in_transform(in_size, x, prev_hs)

  -- Calculate all n gates in one go
  local gs = nn.Sigmoid()(nn.CAddTable()({
    nn.Linear(in_size, 3 * self.capacity)(x),
    nn.Linear(self.capacity, 3 * self.capacity)(prev_h)
  }))

  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates = nn.Reshape(3, self.capacity)(gs)
  local sliced_gates   = nn.SplitTable(2)(reshaped_gates)

  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(2)(sliced_gates))

  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_transform})
  })

  local out_gate         = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end