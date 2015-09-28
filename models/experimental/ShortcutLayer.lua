require("models/BaseEncoderLayer")
require("models/InvTanh")

local ShortcutLayer = torch.class('ShortcutLayer', 'BaseEncoderLayer')

function ShortcutLayer:__init(params, lower_layer)
  self.lower_layer = lower_layer
  params.layer_type = params.layer_type or 'ShortcutLayer'
  params.capacity = lower_layer.capacity
  BaseEncoderLayer.__init(self, params, 1, 1)
end

function ShortcutLayer:create_encoder()
  local x = nn.Identity()()
  local l_s = nn.Identity()()
  local input = nn.Tanh()(nn.Linear(self.in_capacity, self.capacity)(x))
  local s = nn.CAddTable()({l_s, input})

  local m                = nn.gModule({l_s, x}, {s})
  return transfer_data(m)
end

function ShortcutLayer:reset()
  self.start_s:zero()
  self:reset_s()
end

function ShortcutLayer:fp(prev_l, next_l, length, state)
  self.out_s[0]:copy(self.start_s)
  local prev_s
  for i = 1, length do
    local encoder = self:encoder(i)
    prev_s = encoder:forward({self.lower_layer.out_s[i], prev_l.out_s[i]})
    self.out_s[i]:add(prev_s)
  end
  self.start_s:copy(self.out_s[length])
  return 0
end

function ShortcutLayer:bp(prev_l, next_l, length, state)
  for i = length,1,-1 do
    local encoder = self:encoder(i)
    local l_ds, in_ds = unpack(encoder:backward({self.lower_layer.out_s[i], prev_l.out_s[i]}, self.out_ds[i]))
    self.lower_layer.out_ds[i]:add(l_ds)
    prev_l.out_ds[i]:add(in_ds)
  end
end

function ShortcutLayer:setup(batch_size)
  BaseEncoderLayer.setup(self, batch_size)
  self.out_s[0] = transfer_data(torch.zeros(self.batch_size,self.capacity))
  self.out_ds[0] = transfer_data(torch.zeros(self.batch_size,self.capacity))
  self.start_s = transfer_data(torch.zeros(self.batch_size,self.capacity))
end

-- Gated Shortcut (performs nearly equal to Standard Shortcut) --

local GatedShortcutLayer = torch.class('GatedShortcutLayer', 'ShortcutLayer')

function GatedShortcutLayer:__init(params, lower_layer)
  params.layer_type = params.layer_type or 'GatedShortcutLayer'
  ShortcutLayer.__init(self, params, lower_layer)
end

function GatedShortcutLayer:create_encoder()
  local x = nn.Identity()()
  local l_s = nn.Identity()()

  local gates = nn.Sigmoid()(nn.AddConstant(2,true)(nn.CAddTable()({
    nn.Linear(self.in_capacity, 2)(x),
    nn.Linear(self.capacity, 2)(l_s)
  })))
  local shortcut_gate, feedback_gate  = nn.SplitTable(2)(gates):split(2) 
  shortcut_gate = nn.Replicate(self.capacity,2)(shortcut_gate)
  feedback_gate = nn.Replicate(self.capacity,2)(feedback_gate)
  local shortcut = nn.CMulTable()({shortcut_gate, l_s})
  local input    =  nn.CMulTable()({feedback_gate, nn.Tanh()(nn.Linear(self.in_capacity, self.capacity)(x))})
  local s = nn.CAddTable()({shortcut,  input})

  local m                = nn.gModule({l_s, x}, {s})
  return transfer_data(m)
end

-- Weighted Shortcut  (performs much worse than shortcut) --

local WeightedShortcutLayer = torch.class('WeightedShortcutLayer', 'ShortcutLayer')

function WeightedShortcutLayer:__init(params, lower_layer)
  params.layer_type = params.layer_type or 'WeightedShortcutLayer'
  ShortcutLayer.__init(self, params, lower_layer)
end

function WeightedShortcutLayer:create_encoder()
  local x = nn.Identity()()
  local l_s = nn.Identity()()

--  local gates = nn.Sigmoid()(nn.AddConstant(2,true)(nn.CAddTable()({
--    nn.Linear(self.in_capacity, 2)(x),
--    nn.Linear(self.capacity, 2)(l_s)
--  })))
--  local shortcut_gate, feedback_gate  = nn.SplitTable(2)(gates):split(2)
--  shortcut_gate = nn.Replicate(self.capacity,2)(shortcut_gate)
--  feedback_gate = nn.Replicate(self.capacity,2)(feedback_gate)
  local shortcut = nn.Linear(self.capacity, self.capacity)(l_s)
  local input    = nn.Linear(self.in_capacity, self.capacity)(x)
  local s = nn.Tanh()(nn.CAddTable()({shortcut,  input}))

  local m                = nn.gModule({l_s, x}, {s})
  return transfer_data(m)
end
