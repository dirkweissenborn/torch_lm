require('models/RecurrentLayer')
require('models/LSTMLayer')
local util = require("util")
local SelectiveSkipRecurrentLayer = torch.class('SelectiveSkipRecurrentLayer','RecurrentLayer')

function SelectiveSkipRecurrentLayer:__init(params)
  params.layer_type = params.layer_type or 'SelectiveSkipRecurrentLayer'
  self.batch_size = params.batch_size or 1
  BaseEncoderLayer.__init(self, params, 1, 1)
  self.skip = params.skip or 5
end


function SelectiveSkipRecurrentLayer:create_encoder()
  local to_attend    = nn.Identity()() -- 3D: batch x length x in_capacity
  local proj         = nn.MultiLinear(self.in_capacity, 1)(to_attend) -- 3D: batch x length x 1
  self.view1         = nn.View(self.batch_size,-1) -- batch x length
  local attention_w  = nn.SoftMax()(self.view1(proj)) -- batch x length
  self.view2         = nn.View(self.batch_size,-1,1)
  attention_w        = self.view2(attention_w) -- batch x length x 1
  local attention    = nn.MultiMM(true)({to_attend, attention_w}) -- batch x capacity x 1
  attention          = nn.View(-1,self.in_capacity)(attention) -- batch x capacity

  local inputs, next_s =
    create_rnn_encoder(self.type, self.in_capacity, self.capacity, self.depth, self.dropout, attention)
  inputs[1]           = to_attend
  local m         = nn.gModule(inputs, next_s)
  return transfer_data(m)
end


function SelectiveSkipRecurrentLayer:fp(prev_l, next_l, length, state)
  util.replace_table(self.s[0], self.start_s)
  local skip = self.skip
  self.last_length = math.ceil(length/skip)
  self:encoder(self.last_length)
  for i = 1, self.last_length do
    local j = 0
    local off = (i-1)*skip
    while j < skip and j < length-off do
      j = j + 1
      self.to_attend:select(2,j):copy(prev_l.out_s[off+j])
    end
    local to_attend = self.to_attend
    if j < skip then to_attend = self.to_attend:sub(1,-1,1,j,1,-1) end
    local encoder = self:encoder(i)
    local tmp = encoder:forward({to_attend, unpack(self.s[i-1])})
    if type(tmp) == "table" then util.add_table(self.s[i],tmp)
    else self.s[i][1]:add(tmp) end
  end
  util.replace_table(self.start_s, self.s[self.last_length])
  return 0
end

function SelectiveSkipRecurrentLayer:bp(prev_l, next_l, length, state)
  self.split = self.split or nn.SplitTable(2,3)
  local skip = self.skip
  for i = self.last_length,1,-1 do
    local j = 0
    local off = (i-1)*skip
    while j < skip and j < length-(i-1)*skip do
      j = j + 1
      self.to_attend:select(2,j):copy(prev_l.out_s[off+j])
    end
    local to_attend = self.to_attend
    if j < skip then to_attend = self.to_attend:sub(1,-1,1,j,1,-1) end

    local encoder = self:encoder(i)
    local ds
    if self.s_size > 1 then ds = encoder:backward({to_attend, unpack(self.s[i-1])}, self.ds[i])
    else  ds = encoder:backward({to_attend, unpack(self.s[i-1])}, self.ds[i][1]) end
    for j2=off+1,math.min(off+skip,length) do prev_l.out_ds[j2]:add(ds[1]:select(2,j2-off)) end
    for L=1,self.s_size do self.ds[i-1][L]:add(ds[L+1]) end
  end
end

function SelectiveSkipRecurrentLayer:init_encoders(max_num) self:encoder(math.ceil(max_num/self.skip)) end

function SelectiveSkipRecurrentLayer:params()
  local t = RecurrentLayer.params(self)
  t.skip  = self.skip
  t.view1  = self.view1
  t.view2  = self.view2
  return t
end

function SelectiveSkipRecurrentLayer:set_params(t)
  RecurrentLayer.set_params(self,t)
  self.skip = t.skip
  self.view1 = t.view1
  self.view2 = t.view2
end

function SelectiveSkipRecurrentLayer:setup(batch_size)
  self.view1.numElements = self.view1.numElements / self.batch_size * batch_size
  self.view2.numElements = self.view2.numElements / self.batch_size * batch_size
  RecurrentLayer.setup(self, batch_size)
  self.view1.size[1] = batch_size
  self.view2.size[1] = batch_size

  -- create 3D attention (batch_size x attention_length x capacity)
  self.to_attend = transfer_data(torch.Tensor()):resize(batch_size, self.skip, self.in_capacity)
end


------------------ LSTM ----------------------

local SelectiveSkipLSTMLayer = torch.class('SelectiveSkipLSTMLayer', 'SelectiveSkipRecurrentLayer')

function SelectiveSkipLSTMLayer:__init(params)
  params.layer_type = params.layer_type or 'SelectiveSkipLSTMLayer'
  self.depth = params.depth or 1
  self.batch_size = params.batch_size or 1
  self.skip = params.skip or 5
  BaseEncoderLayer.__init(self, params, 2*self.depth, 2*self.depth)
end

function SelectiveSkipLSTMLayer:create_encoder()
  local to_attend    = nn.Identity()() -- 3D: batch x length x in_capacity
  local proj         = nn.MultiLinear(self.in_capacity, 1)(to_attend) -- 3D: batch x length x 1
  self.view1         = nn.View(self.batch_size,-1) -- batch x length
  local attention_w  = nn.SoftMax()(self.view1(proj)) -- batch x length
  self.view2         = nn.View(self.batch_size,-1,1)
  attention_w        = self.view2(attention_w) -- batch x length x 1
  local attention    = nn.MultiMM(true)({to_attend, attention_w}) -- batch x capacity x 1
  attention          = nn.View(-1,self.in_capacity)(attention) -- batch x capacity

  local inputs,next_s = create_lstm_encoder(self.in_capacity, self.capacity, self.depth, self.dropout, attention)
  inputs[1]           = to_attend
  local m             = nn.gModule(inputs, next_s)
  return transfer_data(m)
end
