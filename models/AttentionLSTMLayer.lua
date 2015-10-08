require("models/BaseEncoderLayer")
require("models/LSTMLayer")
require("models/BPDropout")
require("models/MultiLinear")
local AttentionLSTMLayer = torch.class('AttentionLSTMLayer', 'LSTMLayer')

function AttentionLSTMLayer:__init(params)
  params.layer_type = params.layer_type or 'AttentionLSTMLayer'
  self.attention_capacity = params.attention_capacity
  self.batch_size = params.batch_size
  LSTMLayer.__init(self, params)
end

function AttentionLSTMLayer:create_encoder()
  local inputs = {}
  local to_attend = nn.Identity()() -- batch x attention_length x att_capacity
  table.insert(inputs, to_attend) -- attention
  local input = nn.Identity()()
  table.insert(inputs, input) -- x
  input = nn.BPDropout(self.dropout)(input)
  for L = 1,self.depth do
    table.insert(inputs, nn.Identity()()) -- prev_c
    table.insert(inputs, nn.Identity()()) -- prev_h
  end
  
  local next_s        = {}
  local last_out = inputs[2*self.depth+2]

  -- Attention
  -- multilinear has a cache to make sure that this operation is not calculated in every timestep, since it is the same
  local multi_proj   = nn.MultiLinear(self.attention_capacity, self.capacity)(to_attend) -- batch x length x capacity
  
  local last_proj    = nn.ExpandAs(2)({nn.Linear(self.capacity,self.capacity)(last_out), multi_proj})
  local proj         = nn.Tanh()(nn.MultiCAddTable()({multi_proj,last_proj}))

  self.view1         = nn.View(self.batch_size,-1)
  local attention_w  = self.view1(nn.MultiLinear(self.capacity, 1, false)(proj))
  attention_w        = nn.SoftMax()(attention_w) -- batch x attention_length
  self.view2         = nn.View(self.batch_size,-1,1)
  attention_w        = self.view2(attention_w)
  local attention    = nn.MultiMM(true)({to_attend, attention_w}) -- batch x capacity x 1
  attention          = nn.View(-1,self.attention_capacity)(attention) -- batch x capacity
  
  for L = 1,self.depth do
    local prev_c         = inputs[2*L+1]
    local prev_h         = inputs[2*L+2]
    local x
    local in_size = self.in_capacity + self.attention_capacity
    if L == 1 then 
      x = nn.JoinTable(2)({input,attention})
    else 
      x = nn.JoinTable(2)({next_s[2*L-2],input,attention})
      in_size = in_size + self.capacity
    end

    local next_c, next_h = self:lstm(in_size, x, prev_h, prev_c)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
  end

  local m          = nn.gModule(inputs, next_s)
  return transfer_data(m)
end

function AttentionLSTMLayer:fp(prev_l, next_l, length, state)
  util.replace_table(self.s[0], self.start_s)
  self:encoder(length) -- make sure there are enough encoders
  
  -- create 3D (batch_size x attention_length x capacity)
  self.attention = self.attention or 
      transfer_data(torch.Tensor()):resize(self.batch_size, #state.attention.s, self.attention_capacity)
  if self.attention:size(2) < #state.attention.s then
    self.attention:resize(self.batch_size, #state.attention.s, self.attention_capacity)
  end
  for a=1, #state.attention.s do
    self.attention:select(2,a):copy(state.attention.s[a])
  end
  local attention = self.attention:sub(1,-1,1,#state.attention.s,1,-1)
  for i = 1, length do
    local inp = prev_l.out_s[i]
    local lstm = self:encoder(i)
    local tmp = lstm:forward({ attention, inp, unpack(self.s[i-1])})
    util.add_table(self.s[i],tmp)
  end
  util.replace_table(self.start_s, self.s[length])
  return 0
end

function AttentionLSTMLayer:bp(prev_l, next_l, length, state)
  self.split = self.split or nn.SplitTable(2,3)
  local attention = self.attention:sub(1,-1,1,#state.attention.s,1,-1)
  for i = length,1,-1 do
    local inp = prev_l.out_s[i]
    local lstm = self:encoder(i)
    local ds = lstm:backward({attention,inp, unpack(self.s[i-1])}, self.ds[i])
    prev_l.out_ds[i]:add(ds[2])
    for L=1,2*self.depth do self.ds[i-1][L]:add(ds[L+2]) end
    -- attention bp
    local attention_ds = self.split:forward(ds[1])
    util.add_table(state.attention.ds, attention_ds)
  end
end

function AttentionLSTMLayer:params()
  local t = LSTMLayer.params(self)
  t.attention_capacity = self.attention_capacity
  t.view1 = self.view1
  t.view2 = self.view2
  return t
end

function AttentionLSTMLayer:set_params(t)
  LSTMLayer.set_params(self,t)
  self.attention_capacity = t.attention_capacity
  self.view2 = t.view2
  self.view1 = t.view1
end

function AttentionLSTMLayer:setup(batch_size)
  self.view1.numElements = self.view1.numElements / self.batch_size * batch_size
  self.view2.numElements = self.view2.numElements / self.batch_size * batch_size
  LSTMLayer.setup(self, batch_size)
  self.view1.size[1] = batch_size
  self.view2.size[1] = batch_size
  self.attention = nil -- has to be reset
end