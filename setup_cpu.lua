require('nn')

function transfer_data(m) return m:float() end

local function make_deterministic(seed)
  torch.manualSeed(seed)
end

make_deterministic(1)

