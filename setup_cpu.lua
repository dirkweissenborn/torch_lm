require('nn')

function transfer_data(m) return m:double() end

local function make_deterministic(seed)
  torch.manualSeed(seed)
end

make_deterministic(1)

