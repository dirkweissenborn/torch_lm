require("models/autoencoder")
require("setup_cpu")
local data = require("data")

function check_all_grads(auto_encoder, p)
  for k, _ in pairs(auto_encoder.paramx) do
    local b, _, d, dc1, dc2 = check_grad(auto_encoder, k, p)
    if not b then return b, k, d, dc1, dc2 end
  end
  return true
end

function check_grad(auto_encoder, k, p)
  print("Checking " .. k .. "...")
  local paramx = auto_encoder.paramx[k]
  paramx.paramdx:zero()
  local x = paramx.paramx
  
  local prob = p or math.min(1, 100/x:size(1))
  
  local enc = data.convertUTF8("a b. cd.")
  local dec = data.convertUTF8("a b. cd.")

  local state = auto_encoder:new_state(enc,dec,{[4] = 1})
  local length = state.enc.x:size(1)

  auto_encoder.encoder:init_encoders(length)
  auto_encoder.decoder:init_encoders(length)
  auto_encoder:reset()
  auto_encoder:reset_state(state)
  -- forward
  local loss1 = auto_encoder:fp(state, length, length)
  auto_encoder:reset()
  auto_encoder:reset_state(state)
  local loss2 = auto_encoder:fp(state, length, length)
  assert(loss1 == loss2, "Something is wrong in forward pass. Losses are not equal.")
  assert(loss1 ~= 0, "Loss should not be zero.")
  -- backward
  auto_encoder:bp(state, length, length)

  -- compute true gradient:
  local dC = paramx.paramdx

  -- compute numeric approximations to gradient:
  local eps = 1e-4
  for i = 1,dC:size(1) do
    if(torch.bernoulli(prob) == 1) then
      x[i] = x[i] + eps
      auto_encoder:reset()
      auto_encoder:reset_state(state)
      local C1 = auto_encoder:fp(state,length,length)
      x[i] = x[i] - 2 * eps
      auto_encoder:reset()
      auto_encoder:reset_state(state)
      local C2 = auto_encoder:fp(state,length,length)
      x[i] = x[i] + eps
      local dC_est = (C1 - C2) / (2 * eps)

      local diff = math.abs(dC[i] - dC_est)
      if diff > 1.0e-6 then return false, k, diff, dC[i], dC_est end
    end
  end
  
  return true
end

function check_example()
  local init_params = {
    capacity = 1,
    dropout = 0,
    batch_size = 2,
    vocab = data.vocab_utf8
  }
  local enc_layers = {
    [1] = { layer_type = "LSTMLayer", type = "lstm" },
    [2] = { layer_type = "RecurrentLayer", type = "gru"},
    [3] = { layer_type = "LSTMLayer", type = "lstm"}
  }
  local dec_layers = {
    [1] = { layer_type = "RecurrentLayer", type="rnn"  } ,
    [2] = { layer_type = "LSTMLayer", type="lstm" },
    [3] = { layer_type = "LSTMLayer", type="lstm" }
  }
  local encoder = TextAutoEncoder(init_params, enc_layers, dec_layers)
  encoder.encoder:random_init(0.1)
  encoder.decoder:random_init(0.1)
  return check_all_grads(encoder)
end