require('models/BaseEncoderLayer')
require('nn')
require('nngraph')

local AttentionSkipLayer = torch.class('AttentionSkipLayer','BaseEncoderLayer')

function AttentionSkipLayer:__init(underlying, skip)
  self.capacity           = underlying.capacity
  self.dropout            = underlying.dropout
  self.in_capacity        = underlying.in_capacity
  self.init_weight        = underlying.init_weight
  self.name               = underlying.name
  self.s_size             = underlying.s_size
  self.out_index          = underlying.out_index
  self.layer_type         = underlying.layer_type
  self.core_encoder       = underlying.core_encoder
  self.paramx = underlying.paramx
  self.paramdx = underlying.paramdx
  self.stm = underlying.stm
  self.underlying = underlying
  self.skip = skip
  self.attention = self:create_attention()
  self.dummy = { out_s = {}, out_ds = {} }
end

function AttentionSkipLayer:create_encoder() end

function AttentionSkipLayer:create_attention() 
  local x = nn.Identity()()
  local split = { x:split(self.skip) }
  local linear = nn.Linear(self.in_capacity,1)
  local scores = {}
  for i=1, self.skip do
    table.insert(scores, linear(split[i]))
  end
  local join      = nn.JoinTable(2)(scores)
  local sm        = nn.SoftMax()(join)
  local x_join    = nn.JoinTable(3)(x)
  local attending = nn.MixtureTable(2)({sm,x_join})
  local m         = nn.gModule({x}, {attending})
  m:getParameters():uniform(-self.init_weight, self.init_weight)
  return transfer_data(m)
end

function AttentionSkipLayer:reset()
  self.underlying:reset()
  self:reset_s()
end

function AttentionSkipLayer:reset_s()
  self.underlying:reset_s()
  util.zero_table(self.out_s)
  util.zero_table(self.out_ds)
end

function AttentionSkipLayer:fp(prev_l, next_l, length, state)
  self:encoder(length)
  local skip = self.skip
  local off = (state.pos + skip - 1) % skip + 1
  local skip_length = (length-off) / skip + 1
  for i = 1, skip_length do 
    local attention = self:encoder(i)
    local inp = {}
    for j=1, skip do 
      table.insert(inp, prev_l.out_s[math.max(off + (i-2) * skip + j,1)])
    end
    self.dummy.out_s[i] = attention:fp(inp)
  end
  local  loss = self.underlying:fp(self.dummy, nil, skip_length)
  for i = 1, length do
    self.out_s[i]:copy(self.underlying.out_s[math.floor((i-off) / skip)+1])
  end

  return loss
end

function AttentionSkipLayer:bp(prev_l, next_l, length, state)
  util.zero_table(self.dummy.out_ds)
  local skip = self.skip
  local off = (state.pos-length + skip - 1) % skip + 1
  local skip_length = (length-off) / skip + 1
  for i = 1, length do
    local skip_i = math.floor((i-off) / skip)+1
    if skip_i > 0 then
      self.underlying.out_ds[skip_i]:add(self.out_ds[i])
    end
  end
  self.underlying:bp(self.dummy, nil, skip_length)
  for i = 1, skip_length do 
    prev_l.out_ds[off + (i-1) * skip]:add(self.dummy.out_ds[i]) 
  end
  for i = 1, skip_length do
    local attention = self:encoder(i)
    local inp = {}
    for j=1, skip do
      table.insert(inp, prev_l.out_s[math.max(off + (i-2) * skip + j,1)])
    end
    local tmp = attention:bp(inp, self.dummy.out_ds)
    for j=1, skip do
      prev_l.out_ds[math.max(off + (i-2) * skip + j,1)]:add(tmp[i])
    end
  end
end

function AttentionSkipLayer:encoder(i)
  if i > #self.out_s then
    local skip_i = math.floor(i / self.skip) + 1
    self.underlying:encoder(skip_i)
    local atts = util.cloneManyTimes(self.attention, skip_i - #self.attentions)
    for j = 1 , skip_i - #self.dummy.out_s do
      table.insert(self.dummy.out_s, transfer_data(torch.zeros(self.batch_size,self.in_capacity)))
      table.insert(self.dummy.out_ds, transfer_data(torch.zeros(self.batch_size,self.in_capacity)))
      table.insert(self.attentions, atts[j])
    end
    for j = 1, i-#self.out_s do
      table.insert(self.out_s, transfer_data(torch.zeros(self.batch_size,self.capacity)))
      table.insert(self.out_ds, transfer_data(torch.zeros(self.batch_size,self.capacity)))
    end
  end
  return self.attentions[i]
end

function AttentionSkipLayer:setup(batch_size)
  self.underlying:setup(batch_size)
  self.batch_size = batch_size or 1
  self.s = {}
  self.ds = {}
  self.out_s = {}
  self.out_ds = {}
  self.attentions = {}
  --self.dummy.out_s[0] = transfer_data(torch.zeros(batch_size,self.in_capacity))
  --self.dummy.out_ds[0] = transfer_data(torch.zeros(batch_size,self.in_capacity))
end

function AttentionSkipLayer:params()
  local t = self.underlying:params()
  t.skip  = self.skip
  t.attention  = self.attention
  return t
end

function AttentionSkipLayer:set_params(t)
  underlying:set_params(t)
  self.skip = t.skip
  self.attention = t.attention
end

function AttentionSkipLayer:disable_training()
  self.underlying:disable_training()
end

function AttentionSkipLayer:enable_training()
  self.underlying:enable_training()
end