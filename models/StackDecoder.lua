require('models/StackEncoder')
require('models/PredictionLayer')
--require('models/ShortcutLayer')
require('pl')
local data = require('data')

local StackDecoder = torch.class('StackDecoder','StackEncoder')

function StackDecoder:__init(params)
  local ls = params.layers
  if ls then
    if not ls[#ls] or ls[#ls].layer_type ~= "PredictionLayer" then
      if params.shortcut then
        local layers =#ls
        for j = layers-1,params.shortcut_begin or 2,-1 do
          ls[#ls+1] = { 
            layer_type = params.shortcut,
            from = j,
            condition = ls[j].condition,
            skip = ls[j].skip
          }
        end
      end
      --ls[#ls+1] = { layer_type = "TanhLayer", capacity = params.embedding_size }
      ls[#ls+1] = { layer_type = "PredictionLayer", repeats = params.repeats }
    end
  end
  StackEncoder.__init(self, params)
end

function StackDecoder:new_layer(index, params)
  local l = StackEncoder.new_layer(self, index, params)
  if l then return l
  elseif params.layer_type == 'PredictionLayer' then
    params.capacity = tablex.size(self.vocab)
    params.dropout  = self.dropout or 0
    return PredictionLayer(params)
  --[[elseif params.layer_type == 'ShortcutLayer' then
    local shortcut_index = params.from or 1
    if params.condition or params.skip then
      return ShortcutLayer(params, self.layers[shortcut_index].underlying)
    else
      return ShortcutLayer(params, self.layers[shortcut_index])
    end
  elseif params.layer_type == 'GatedShortcutLayer' then
    local shortcut_index = params.from or 1
    if params.condition or params.skip then
      return GatedShortcutLayer(params, self.layers[shortcut_index].underlying)
    else
      return GatedShortcutLayer(params, self.layers[shortcut_index])
    end
  elseif params.layer_type == 'GatedAvgShortcutLayer' then
    local shortcut_index = params.from or 1
    if params.condition or params.skip then
      return GatedAvgShortcutLayer(params, self.layers[shortcut_index].underlying)
    else
      return GatedAvgShortcutLayer(params, self.layers[shortcut_index])
    end
  elseif params.layer_type == 'WeightedShortcutLayer' then
    local shortcut_index = params.from or 1
    if params.condition or params.skip then
      return WeightedShortcutLayer(params, self.layers[shortcut_index].underlying)
    else
      return WeightedShortcutLayer(params, self.layers[shortcut_index])
    end--]]
  end
end

--[[function StackDecoder:run(state, length)
  length = length or 1
  local loss = 0
  state.pos = 0
  while state.pos <= state.x:size(1)-length do
    local l = self:fp(state,length)
    loss = loss + l
  end
  if state.pos < state.x:size(1) then
    loss = loss + self:fp(state, state.x:size(1) - state.pos)
  end

  return loss / state.x:size(1)
end --]]