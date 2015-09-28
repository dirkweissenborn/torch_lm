require("models/BaseEncoderLayer")
require("models/OneHot")
require("data")

local OneHotLayer = torch.class('OneHotLayer', 'BaseEncoderLayer')

function OneHotLayer:__init(params)
  params.layer_type = params.layer_type or 'OneHotLayer'
  params.capacity = params.in_capacity
  BaseEncoderLayer.__init(self, params, 1, 1)
end

function OneHotLayer:create_encoder()
  local x = nn.Identity()()
  local one_hot = nn.OneHot(self.capacity)(x)
  return transfer_data(nn.gModule({x},{one_hot}))
end

function OneHotLayer:fp(prev_l, next_l, length, state)
  for i = 1, length do
    local inp = state.x[(state.pos + i - 1) % state.x:size(1) + 1]
    local enc = self:encoder(i)
    local tmp = enc:forward(inp)
    self.out_s[i]:add(tmp)
  end
  return 0
end

function OneHotLayer:bp(prev_l, next_l, length, state)
end