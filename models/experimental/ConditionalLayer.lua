require('models/BaseEncoderLayer')
require('nn')
require('nngraph')
local util = require('util')
-- CONDITIONS --

local conditions = {
  ["non_alpha_numeric_condition"] = function(x) return not convertNumToUTF8(x):match("%w") end,
  ["punctuation_condition"] = function(x) return not convertNumToUTF8(x):match("%p") end,
  ["space_condition"] = function(x) return not convertNumToUTF8(x):match("%s") end
}

local ConditionalLayer = torch.class('ConditionalLayer','BaseEncoderLayer')

function ConditionalLayer:__init(underlying, condition)
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
  self.condition = condition
  self.f = conditions[condition]
  self.dummy = { out_s = {}, out_ds = {} }
end

function ConditionalLayer:create_encoder() end

function ConditionalLayer:reset() 
  self.underlying:reset()
  self:reset_s()
end

function ConditionalLayer:reset_s() 
  self.underlying:reset_s()
  util.zero_table(self.out_s)
  util.zero_table(self.out_ds)
end

function ConditionalLayer:fp(prev_l, next_l, length, state)
  local p_cond = prev_l == nil or prev_l.condition ~= self.condition
  local n_cond = next_l == nil or next_l.condition ~= self.condition
  --make sure all states are available
  self:encoder(length)
  
  local loss
  -- this layer only receives input at certain points; we also need to synchronize batches at this point (tricky)
  util.zero_table(self.dummy.out_s)
  local batch_index = {}
  for b = 1, self.batch_size do batch_index[b] = 0 end
  for i = 1, length do
    for b = 1, self.batch_size do
      local x = state.x[state.pos+i][b]
      if self.f(x) then
        batch_index[b] = batch_index[b] + 1
        while p_cond and #self.dummy.out_s < batch_index[b] do
          table.insert(self.dummy.out_s, transfer_data(torch.zeros(self.batch_size,self.in_capacity)))
          table.insert(self.dummy.out_ds, transfer_data(torch.zeros(self.batch_size,self.in_capacity)))
        end
        if p_cond then self.dummy.out_s[batch_index[b]][b] = prev_l.out_s[i][b] end
      end
    end
  end
  
  local max_index = 0
  for b = 1, self.batch_size do max_index = math.max(max_index, batch_index[b]) end

  -- init underlying encoders
  if #self.underlying.encoders < max_index then
    self.underlying:encoder(math.ceil(1.5*max_index)) -- create underlying encoders
  end
  
  if p_cond then
    loss = self.underlying:fp(self.dummy, nil, max_index)
  else 
    loss = self.underlying:fp(prev_l.underlying, nil, max_index)
  end
  
  if n_cond then
    -- create correct output for the next layer
    local batch_index = {}
    for b = 1, self.batch_size do batch_index[b] = 0 end
    for i = 1, length do
      for b = 1, self.batch_size do
        local x = state.x[state.pos+i][b]
        if self.f(x) then batch_index[b] = batch_index[b] + 1 end
        self.out_s[i][b]:add(self.underlying.out_s[batch_index[b]][b])
      end
    end
  end
  
  return loss
end

function ConditionalLayer:bp(prev_l, next_l, length, state)
  local p_cond = prev_l == nil or prev_l.condition ~= self.condition
  local n_cond = next_l == nil or next_l.condition ~= self.condition

  -- this layer only receives input at certain points; we also need to synchronize batches at this point (tricky)
  local batch_index = {}
  for b = 1, self.batch_size do batch_index[b] = 0 end
  -- build input & derivatives
  for i = 1, length do
    for b = 1, self.batch_size do
      local x = state.x[state.pos-length+i][b]
      if self.f(x) then batch_index[b] = batch_index[b] + 1 end
      if n_cond and batch_index[b] > 0 then
        self.underlying.out_ds[batch_index[b]][b]:add(self.out_ds[i][b])
      end
    end
  end
  local max_index = 0
  for b = 1, self.batch_size do max_index = math.max(max_index, batch_index[b]) end

  if p_cond then
    util.zero_table(self.dummy.out_ds)
    self.underlying:bp(self.dummy, nil, max_index)
    for i= length, 1, -1 do
      for b = 1, self.batch_size do
        local x = state.x[state.pos-length+i][b]
        if self.f(x) then
          prev_l.out_ds[i][b]:add(self.dummy.out_ds[batch_index[b] ][b])
          batch_index[b] = batch_index[b] - 1
        end
      end
    end
  else self.underlying:bp(prev_l.underlying, nil,  max_index) end
end

function ConditionalLayer:encoder(i)
  if i > #self.out_s then
    for j = 1, i-#self.out_s do
      table.insert(self.out_s, transfer_data(torch.zeros(self.batch_size,self.capacity)))
      table.insert(self.out_ds, transfer_data(torch.zeros(self.batch_size,self.capacity)))
    end
  end
  return nil
end

function ConditionalLayer:setup(batch_size)
  self.underlying:setup(batch_size)
  self.batch_size = batch_size or 1
  self.s = {}
  self.ds = {}
  self.out_s = {}
  self.out_ds = {}
  self.dummy.out_s[0] = transfer_data(torch.zeros(batch_size,self.in_capacity))
  self.dummy.out_ds[0] = transfer_data(torch.zeros(batch_size,self.in_capacity))
end

function ConditionalLayer:params()
  local t = self.underlying:params()
  t.condition  = self.condition
  return t
end

function ConditionalLayer:set_params(t)
  underlying:set_params(t)
  self.condition = t.condition
end

function ConditionalLayer:disable_training()
  self.underlying:disable_training()
end

function ConditionalLayer:enable_training()
  self.underlying:enable_training()
end