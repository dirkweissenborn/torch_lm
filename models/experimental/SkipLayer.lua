require('models/BaseEncoderLayer')
require('nn')
require('nngraph')

local SkipLayer = torch.class('SkipLayer','BaseEncoderLayer')

function SkipLayer:__init(underlying, skip)
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
  self.dummy = { out_s = {}, out_ds = {} }
end

function SkipLayer:create_encoder() end

function SkipLayer:reset()
  self.underlying:reset()
  self:reset_s()
end

function SkipLayer:reset_s()
  self.underlying:reset_s()
  util.zero_table(self.out_s)
  util.zero_table(self.out_ds)
end

function SkipLayer:fp(prev_l, next_l, length, state)
  self:encoder(length)
  local skip = self.skip
  local off = (state.pos + skip - 1) % skip + 1
  local skip_length = (length-off) / skip + 1
  for i = 1, skip_length do self.dummy.out_s[i]:copy(prev_l.out_s[off + (i-1) * skip]) end
  local  loss = self.underlying:fp(self.dummy, nil, skip_length)
  for i = 1, length do 
    self.out_s[i]:copy(self.underlying.out_s[math.floor((i-off) / skip)+1])
  end

  return loss
end

function SkipLayer:bp(prev_l, next_l, length, state)
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
  for i = 1, skip_length do prev_l.out_ds[off + (i-1) * skip]:add(self.dummy.out_ds[i]) end
end

function SkipLayer:encoder(i)
  if i > #self.out_s then
    local skip_i = math.floor(i / self.skip) + 1
    self.underlying:encoder(skip_i)
    for j = 1 , skip_i - #self.dummy.out_s do
      table.insert(self.dummy.out_s, transfer_data(torch.zeros(self.batch_size,self.in_capacity)))
      table.insert(self.dummy.out_ds, transfer_data(torch.zeros(self.batch_size,self.in_capacity)))
    end
    for j = 1, i-#self.out_s do
      table.insert(self.out_s, transfer_data(torch.zeros(self.batch_size,self.capacity)))
      table.insert(self.out_ds, transfer_data(torch.zeros(self.batch_size,self.capacity)))
    end
  end
  return nil
end

function SkipLayer:setup(batch_size)
  self.underlying:setup(batch_size)
  self.batch_size = batch_size or 1
  self.s = {}
  self.ds = {}
  self.out_s = {}
  self.out_ds = {}
  --self.dummy.out_s[0] = transfer_data(torch.zeros(batch_size,self.in_capacity))
  --self.dummy.out_ds[0] = transfer_data(torch.zeros(batch_size,self.in_capacity))
end

function SkipLayer:params()
  local t = self.underlying:params()
  t.skip  = self.skip
  return t
end

function SkipLayer:set_params(t)
  underlying:set_params(t)
  self.skip = t.skip
end

function SkipLayer:disable_training()
  self.underlying:disable_training()
end

function SkipLayer:enable_training()
  self.underlying:enable_training()
end