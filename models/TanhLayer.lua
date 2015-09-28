require("models/BaseEncoderLayer")
require("models/InvTanh")

local TanhLayer = torch.class('TanhLayer', 'BaseEncoderLayer')

function TanhLayer:__init(params)
  params.layer_type = params.layer_type or 'TanhLayer'
  BaseEncoderLayer.__init(self, params, 1, 1)
end

function TanhLayer:create_encoder()
  local x = nn.Identity()()
  local out = nn.Tanh()(nn.Linear(self.in_capacity, self.capacity)(x))
  local m                = nn.gModule({x}, {out})
  return transfer_data(m)
end

function TanhLayer:fp(prev_l, next_l, length, state)
  for i = 1, length do
    local encoder = self:encoder(i)
    local out = encoder:forward(prev_l.out_s[i])
    self.out_s[i]:add(out)
  end
  return 0
end

function TanhLayer:bp(prev_l, next_l, length, state)
  for i = length,1,-1 do
    local encoder = self:encoder(i)
    local in_ds = encoder:backward(prev_l.out_s[i], self.out_ds[i])
    prev_l.out_ds[i]:add(in_ds)
  end
end