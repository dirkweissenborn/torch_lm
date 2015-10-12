require('nn')
require('nngraph')
local util = require('util')

local BaseEncoderLayer = torch.class('BaseEncoderLayer')

function BaseEncoderLayer:__init(params, s_size, out_index)
  self.capacity           = params.capacity
  self.dropout            = params.dropout
  self.in_capacity        = params.in_capacity
  self.name               = params.name or ""
  self.s_size             = s_size
  self.out_index          = out_index
  self.layer_type         = params.layer_type or self.__typename
  self.core_encoder       = self:create_encoder()
  self.train              = true
end

function BaseEncoderLayer:create_encoder()
  error("Not implemented!")
end

function BaseEncoderLayer:reset()
  self:reset_s()
end

function BaseEncoderLayer:reset_s()
  for _,v in pairs(self.s) do util.zero_table(v) end
  for _,v in pairs(self.ds) do util.zero_table(v) end
end

function BaseEncoderLayer:fp(prev_l, next_l, length, state)
  error("Not implemented!")
end

function BaseEncoderLayer:bp(prev_l, next_l, length, state)
  error("Not implemented!")
end

function BaseEncoderLayer:encoder(i)
  if i > #self.out_s then
    local new_encs
    if self.core_encoder then
      new_encs = util.cloneManyTimes(self.core_encoder,math.ceil(i-#self.out_s))
    end
    for j = 1, i-#self.out_s do
      if self.core_encoder then table.insert(self.encoders, new_encs[j]) end
      local s = {}
      local ds = {}
      for d = 1, self.s_size do
        s[d] = transfer_data(torch.zeros(self.batch_size,self.capacity))
        ds[d] = transfer_data(torch.zeros(self.batch_size,self.capacity))
      end
      table.insert(self.s, s)
      table.insert(self.ds, ds)
      table.insert(self.out_s, s[self.out_index])
      table.insert(self.out_ds, ds[self.out_index])
    end
  end
  return self.encoders[i]
end

function BaseEncoderLayer:setup(batch_size)
  self.batch_size = batch_size or 1
  self.encoders = {}
  self.s = {}
  self.ds = {}
  self.out_s = {}
  self.out_ds = {}
end

function BaseEncoderLayer:params()
  local t = {}
  t.core_encoder = self.core_encoder
  t.capacity     = self.capacity
  t.name         = self.name
  t.s_size       = self.s_size
  t.out_index    = self.out_index
  t.layer_type   = self.layer_type
  t.in_capacity  = self.in_capacity
  return t
end

function BaseEncoderLayer:set_params(t)
  self.core_encoder = t.core_encoder
  self.capacity     = t.capacity
  self.name         = t.name
  self.s_size       = t.s_size
  self.out_index    = t.out_index
  self.layer_type   = t.layer_type
  self.in_capacity  = t.in_capacity
  collectgarbage()
end

function BaseEncoderLayer:save(f)
  torch.save(f,self:params())
end

function BaseEncoderLayer:load(f)
  local t = torch.load(f)
  self:set_params(t)
end

function BaseEncoderLayer:disable_training()
  self.train = false
  if self.core_encoder then
    self.core_encoder:evaluate()
    for _,e in pairs(self.encoders) do e:evaluate() end
  end  
end

function BaseEncoderLayer:enable_training()
  self.train = true
  if self.core_encoder then
    self.core_encoder:training()
    for _,e in pairs(self.encoders) do e:training() end
  end
end

function BaseEncoderLayer:init_encoders(max_num) 
  self:encoder(max_num) 
end

function BaseEncoderLayer:networks(networks)
  networks = networks or {}
  table.insert(networks,self.core_encoder)
  return networks
end