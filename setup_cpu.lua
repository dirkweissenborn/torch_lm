require('nn')

function transfer_data(m) return m end --:float() end

local function make_deterministic(seed)
  torch.manualSeed(seed)
  torch.zeros(1, 1):uniform()
end

make_deterministic(1)

