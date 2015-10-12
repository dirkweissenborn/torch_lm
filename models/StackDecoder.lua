require('models/StackEncoder')
require('models/PredictionLayer')
require('models/JoinedPredictionLayer')
require('pl')
local data = require('data')

local StackDecoder = torch.class('StackDecoder','StackEncoder')

function StackDecoder:__init(params)
  local ls = params.layers
  if ls then
    if not ls[#ls] or not stringx.endswith(ls[#ls].layer_type, "PredictionLayer") then
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
      print("Added JoinedPredictionLayer to decoder, because no prediction layer was defined.")
      ls[#ls+1] = { layer_type = "JoinedPredictionLayer" }
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
  elseif params.layer_type == 'JoinedPredictionLayer' then
    params.capacity = tablex.size(self.vocab)
    params.dropout  = self.dropout or 0
    return JoinedPredictionLayer(params, tablex.copy(self.layers))
  end
end