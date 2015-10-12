require('models/LSTMLayer')
require('models/AttentionLSTMLayer')
--require('models/RepeatLSTMLayer')
require('models/DepthGatedLSTMLayer')
require('models/GatedFeedbackLSTMLayer')
require('models/RecurrentLayer')
require('models/LookupLayer')
require('models/SelectiveSkipLayer')
require('models/SelectiveSkipRecurrentLayer')
require('models/OneHotLayer')
require('models/TanhLayer')
local data = require('data')

local StackEncoder = torch.class('StackEncoder')

function StackEncoder:new_layer(index, params)
  params.capacity = params.capacity or self:layer_capacity(index)
  params.in_capacity = params.in_capacity or self:layer_capacity(index-1)
  params.name = params.name or (tostring(index) .. "-" .. params.layer_type)
  
  local l 
  if params.layer_type == 'LookupLayer' then
    l = LookupLayer(params)
  elseif params.layer_type == 'OneHotLayer' then  --not efficient
    l = OneHotLayer(params)
  elseif params.layer_type == 'RecurrentLayer' then
    l =  RecurrentLayer(params)
  elseif params.layer_type == 'TanhLayer' then
    l =  TanhLayer(params)
  elseif params.layer_type == 'LSTMLayer' then
    l = LSTMLayer(params)
  elseif params.layer_type == 'AttentionLSTMLayer' then
    params.batch_size = self.batch_size
    l = AttentionLSTMLayer(params)
  elseif params.layer_type == 'SelectiveSkipLayer' then
    params.batch_size = self.batch_size
    l = SelectiveSkipLayer(params)
  elseif params.layer_type == 'SelectiveSkipRecurrentLayer' then
    params.batch_size = self.batch_size
    l = SelectiveSkipRecurrentLayer(params)
  elseif params.layer_type == 'SelectiveSkipLSTMLayer' then
    params.batch_size = self.batch_size
    l = SelectiveSkipLSTMLayer(params)
  elseif params.layer_type == 'GatedFeedbackLSTMLayer' then
    l = GatedFeedbackLSTMLayer(params)
  elseif params.layer_type == 'RevGatedFeedbackLSTMLayer' then
    l = RevGatedFeedbackLSTMLayer(params)
  elseif params.layer_type == 'DepthGatedLSTMLayer' then
    l = DepthGatedLSTMLayer(params)
  end
  
  return l
end

function StackEncoder:__init(params) 
  params = params or {}
  self.capacity = params.capacity or 1
  self.embedding_size = params.embedding_size
  self.vocab = params.vocab
  self.lookup = params.lookup
  self.scaling = params.scaling or 1
  self.batch_size = params.batch_size or 1
  self.dropout = params.dropout or 0
  self.layers = {}
  self.train = true

  if params.layers then
    local lp = { layer_type = "LookupLayer" }
    if params.noise_variance and params.noise_variance > 0 then lp.noise_variance = params.noise_variance end
    if params.flip_prob and params.flip_prob > 0 then lp.flip_prob = params.flip_prob end
    if not self.lookup then lp.layer_type = "OneHotLayer" end
    params.layers[0] = lp
    
    for k=0,#params.layers do
      local v = params.layers[k]
      if not v.dropout then
        if self.lookup and k == 1 then v.dropout = self.dropout or 0 --apply dropout only after input
        else v.dropout = 0 end
      end

      self.layers[k] = self:new_layer(k, v)
      assert(self.layers[k], v.layer_type .. " is an unknown LayerType.")
      self.layers[k]:setup(self.batch_size)
    end
  end
end

function StackEncoder:init_encoders(max_num)
  for _,l in pairs(self.layers) do l:init_encoders(max_num) end
end

function StackEncoder:random_init(init_weight)
  init_weight = init_weight or 0.08
  for _, n in pairs(self:networks()) do
    n:getParameters():uniform(-init_weight,init_weight)
  end
end

function StackEncoder:layer_capacity(index)
  if(index < 0) then return tablex.size(self.vocab)
  elseif self.layers[index] then return self.layers[index].capacity
  elseif index == 0 then return self.embedding_size
  else
    local scale = math.pow(self.scaling, math.max(index-1,0))
    return math.ceil(scale * self.capacity)
  end
end

function StackEncoder:new_state(str, old_state)
  local encode = data.convertChars(str,self.vocab)
  local s = old_state or {}
  s.pos = 0
  s.x = data.replicate(encode, self.batch_size, s.x)
  local ydata = encode:clone()
  if s.x:size(1) > 1 then
    ydata:sub(1,-2):copy(encode:sub(2,-1))
  end
  ydata[-1] = self.vocab['<eos>']
  s.y = data.replicate(ydata, self.batch_size, s.y)
  return s
end

function StackEncoder:reset_state(s) s.pos = 0 end

function StackEncoder:reset() for _, l in pairs(self.layers) do l:reset() end end

function StackEncoder:reset_s() 
  for _, l in pairs(self.layers) do l:reset_s() end 
end

function StackEncoder:fp(state, length, no_reset)
  if not no_reset then self:reset_s() end
  local start_l = 1
  if self.layers[0] then start_l = 0 end

  local loss = 0
  local last_length = length
  for j=start_l, #self.layers do
    local prev_l = self.layers[j-1]
    local next_l = self.layers[j+1]
    local l = self.layers[j]
    if l.layer_type == "JoinedPredictionLayer" then last_length = length
    elseif j > start_l then last_length = prev_l.last_length or last_length end
    loss = loss + l:fp(prev_l, next_l, last_length, state)
  end
  
  state.pos = (state.pos + length - 1) % state.x:size(1) + 1
  return loss
end

function StackEncoder:bp(state, length)
  local start_l = 1
  if self.layers[0] then start_l = 0 end
  state.pos = (state.pos - length - 1 + state.x:size(1)) % state.x:size(1) + 1

  local last_lengths = {[start_l]=length}
  for j=start_l+1, #self.layers do
    local prev_l = self.layers[j-1]
    local l = self.layers[j]
    if l.layer_type == "JoinedPredictionLayer" then last_lengths[j] = length
    elseif j > start_l then last_lengths[j] = prev_l.last_length or last_lengths[j-1] end
  end
  for j=#self.layers,start_l,-1 do
    local prev_l = self.layers[j-1]
    local next_l = self.layers[j+1]
    local l = self.layers[j]
    l:bp(prev_l, next_l, last_lengths[j], state)
  end

  state.pos = (state.pos + length - 1) % state.x:size(1) + 1
end

function StackEncoder:setup(batch_size)
  self.batch_size = batch_size or self.batch_size
  self.tmp_s = {}
  for _, l in pairs(self.layers) do l:setup(self.batch_size) end
end

function StackEncoder:params()
  local t = {}
  t.capacity = self.capacity
  t.scaling  = self.scaling
  t.dropout  = self.dropout
  t.vocab    = self.vocab
  t.lookup   = self.lookup
  t.embedding_size   = self.embedding_size

  t.layers = {}
  for k,v in pairs(self.layers) do
    t.layers[k] = v:params()
  end
  return t
end

function StackEncoder:set_params(t)
  self.capacity = t.capacity
  self.scaling  = t.scaling
  self.dropout  = t.dropout
  self.vocab    = t.vocab
  self.lookup   = t.lookup
  self.embedding_size   = t.embedding_size
  self.layers   = {}
  for k=0, #t.layers do
    local l = t.layers[k]
    if l then
      --if stringx.endswith(l.layer_type, 'ShortcutLayer') then l.from = #t.layers - k end -- #t.layers
      self.layers[k] = self:new_layer(k, l)
      self.layers[k]:set_params(l)
    end
  end
  self:setup()
end

function StackEncoder:save(f)
  torch.save(f,self:params())
end

function StackEncoder:load(f)
  self:set_params(torch.load(f))
end

function StackEncoder:disable_training()
  self.train = false
  for _, l in pairs(self.layers) do l:disable_training() end
end

function StackEncoder:enable_training()
  self.train = true
  for _, l in pairs(self.layers) do l:enable_training() end
end

function StackEncoder:networks(networks)
  networks = networks or {}
  for _, l in pairs(self.layers) do l:networks(networks) end
  return networks
end