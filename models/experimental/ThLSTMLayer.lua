require("models/BaseEncoderLayer")
require("models/BPDropout")
require("models/LSTMLayer")
local ThLSTMLayer = torch.class('ThLSTMLayer', 'LSTMLayer')

function ThLSTMLayer:__init(params)
  params.layer_type = params.layer_type or 'ThLSTMLayer'
  self.inner_depth = params.inner_depth
  LSTMLayer.__init(self, params)
end

function ThLSTMLayer:create_encoder()
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
    local next_c = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c }),
      nn.CMulTable()({in_gate,     in_transform})
    })
    local out = nn.Tanh()(next_c)

    lins[L] ={}
    for j=1, self.inner_depth-1 do
      local l = nn.Linear(self.capacity, self.capacity)
      lins[L][j] = l
      local inp = nn.Tanh()(l(out))
      next_c = nn.CAddTable()({next_c, inp})
      out = nn.Tanh()(next_c)
    end
    local next_h = nn.CMulTable()({out_gate, out})

    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
  end

  local m          = nn.gModule(inputs, next_s)
  m = transfer_data(m)
  --for L = 1,self.depth do
  --  for j=2, self.inner_depth-1 do lins[L][j]:share(lins[L][1],'weight','bias', 'gradWeight', 'gradBias') end
  --end
  return m
end

function ThLSTMLayer:params()
  local t = LSTMLayer.params(self)
  t.inner_depth       = self.inner_depth
  return t
end

function ThLSTMLayer:set_params(t)
  LSTMLayer.set_params(self,t)
  self.inner_depth    = t.inner_depth
end