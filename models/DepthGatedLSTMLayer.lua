require("models/LSTMLayer")

local DepthGatedLSTMLayer = torch.class('DepthGatedLSTMLayer', 'LSTMLayer')

function DepthGatedLSTMLayer:__init(params)
  params.layer_type = params.layer_type or 'DepthGatedLSTMLayer'
  LSTMLayer.__init(self, params)
end

function DepthGatedLSTMLayer:create_encoder()
  local inputs   = {}
  local next_s   = {}
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

    if L > 1 then
      local depth_gate = nn.Sigmoid()(nn.CAddTable()({
        nn.Linear(self.capacity, self.capacity)(x),
        nn.CMul(self.capacity)(next_s[2*L-3]),
        nn.CMul(self.capacity)(prev_c)
      }))
      next_c = nn.CAddTable()({
        nn.CMulTable()({depth_gate, next_s[2*L-3]}),
        next_c
      })
    else
      local depth_gate = nn.Sigmoid()(nn.CAddTable()({
        nn.Linear(self.in_capacity, self.capacity)(x),
        nn.CMul(self.capacity)(prev_c)
      }))
      next_c = nn.CAddTable()({
        nn.CMulTable()({depth_gate, nn.Linear(self.in_capacity, self.capacity)(x)}),
        next_c
      })
    end
    
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
  end

  local m          = nn.gModule(inputs, next_s)
  return transfer_data(m)
end
