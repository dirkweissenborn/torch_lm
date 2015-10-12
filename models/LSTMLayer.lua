require("models/RecurrentLayer")
require("models/BPDropout")
local LSTMLayer = torch.class('LSTMLayer', 'RecurrentLayer')

function LSTMLayer:__init(params)
  params.layer_type = params.layer_type or 'LSTMLayer'
  self.depth = params.depth or 1
  BaseEncoderLayer.__init(self, params, 2*self.depth, 2*self.depth)
end

function LSTMLayer:create_encoder()
  local inputs,next_s = create_lstm_encoder(self.in_capacity, self.capacity, self.depth, self.dropout)
  local m             = nn.gModule(inputs, next_s)
  return transfer_data(m)
end

function create_lstm_encoder(in_capacity, capacity, depth, dropout, input)
  dropout = dropout or 0
  local inputs = {}
  local next_s        = {}
  table.insert(inputs, input or nn.Identity()()) -- x
  for L = 1,depth do
    local prev_c         = nn.Identity()()
    local prev_h         = nn.Identity()()
    table.insert(inputs, prev_c)
    table.insert(inputs, prev_h)
    local x
    if L == 1 then x = nn.BPDropout(dropout)(inputs[1])
    else x = next_s[2*L-2] end

    local in_size = in_capacity
    if L > 1 then in_size = capacity end
    local next_c, next_h = lstm(in_size, capacity, x, prev_h, prev_c)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
  end

  return inputs, next_s
end

function lstm(in_size, out_size, x, prev_h, prev_c)
  -- evaluate the input sums at once for efficiency
  local i2h = nn.Linear(in_size, 4 * out_size)(x)
  local h2h = nn.Linear(out_size, 4 * out_size)(prev_h)
  local all_input_sums = nn.CAddTable()({i2h, h2h})

  local reshaped = nn.Reshape(4, out_size)(all_input_sums)
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
