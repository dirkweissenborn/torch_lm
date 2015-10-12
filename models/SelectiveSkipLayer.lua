require('models/BaseEncoderLayer')
require('nn')
require('nngraph')

local SelectiveSkipLayer = torch.class('SelectiveSkipLayer','BaseEncoderLayer')

function SelectiveSkipLayer:__init(params)
  params.layer_type = params.layer_type or 'SelectiveSkipLayer'
  self.batch_size = params.batch_size or 1
  BaseEncoderLayer.__init(self, params, 1, 1)
  self.capacity = self.in_capacity
  self.skip = params.skip or 5
end


function SelectiveSkipLayer:create_encoder()
  local to_attend    = nn.Identity()() -- 3D: batch x length x capacity
  local proj         = nn.MultiLinear(self.in_capacity, 1)(to_attend) -- 3D: batch x length x 1
  self.view1         = nn.View(self.batch_size,-1) -- batch x length
  local attention_w  = nn.SoftMax()(self.view1(proj)) -- batch x length
  self.view2         = nn.View(self.batch_size,-1,1)
  attention_w        = self.view2(attention_w) -- batch x length x 1
  local attention    = nn.MultiMM(true)({to_attend, attention_w}) -- batch x capacity x 1
  attention          = nn.View(-1,self.capacity)(attention) -- batch x capacity

  local m         = nn.gModule({to_attend}, {attention})
  return transfer_data(m)
end


function SelectiveSkipLayer:fp(prev_l, next_l, length, state)
  local skip = self.skip
  self.last_length = math.ceil(length/skip)
  self:encoder(self.last_length)
  for i = 1, self.last_length do
    local j = 0
    local off = (i-1)*skip
    while j < skip and j < length-off do
      j = j + 1
      self.attention:select(2,j):copy(prev_l.out_s[off+j])
    end
    local to_attend = self.attention
    if j < skip then to_attend = self.attention:sub(1,-1,1,j,1,-1) end
    local encoder = self:encoder(i)
    self.out_s[i] = encoder:forward(to_attend)
  end
  return 0
end

function SelectiveSkipLayer:bp(prev_l, next_l, length, state)
  self.split = self.split or nn.SplitTable(2,3)
  local skip = self.skip
  for i = self.last_length,1,-1 do
    local j = 0
    local off = (i-1)*skip
    while j < skip and j < length-(i-1)*skip do
      j = j + 1
      self.attention:select(2,j):copy(prev_l.out_s[off+j])
    end
    local to_attend = self.attention
    if j < skip then to_attend = self.attention:sub(1,-1,1,j,1,-1) end
    local ds = self:encoder(i):backward(to_attend, self.out_ds[i])

    for j2=off+1,math.min(off+skip,length) do prev_l.out_ds[j2]:add(ds:select(2,j2-off)) end
  end
end

function SelectiveSkipLayer:init_encoders(max_num) self:encoder(math.ceil(max_num/self.skip)) end

function SelectiveSkipLayer:params()
  local t = BaseEncoderLayer.params(self)
  t.skip  = self.skip
  t.view1  = self.view1
  t.view2  = self.view2
  return t
end

function SelectiveSkipLayer:set_params(t)
  BaseEncoderLayer.set_params(self,t)
  self.skip = t.skip
  self.view1 = t.view1
  self.view2 = t.view2
end

function SelectiveSkipLayer:setup(batch_size)
  self.view1.numElements = self.view1.numElements / self.batch_size * batch_size
  self.view2.numElements = self.view2.numElements / self.batch_size * batch_size
  BaseEncoderLayer.setup(self, batch_size)
  self.view1.size[1] = batch_size
  self.view2.size[1] = batch_size

  -- create 3D attention (batch_size x attention_length x capacity)
  self.attention = transfer_data(torch.Tensor()):resize(batch_size, self.skip, self.in_capacity)
end
