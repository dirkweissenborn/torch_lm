require("fbcunn")

use_cuda = true

local function make_deterministic(seed)
  torch.manualSeed(seed)
  cutorch.manualSeed(seed)
end

function init_gpu(gpuidx)
  print(string.format("Using %s-th gpu", gpuidx))
  cutorch.setDevice(gpuidx)
  make_deterministic(1)
end

function transfer_data(m) return m:cuda() end


