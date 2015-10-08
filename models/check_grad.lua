require("models/AttentionEncoderDecoder")
require("setup_cpu")
local data = require("data")

function check_all_grads(encdec, p)
  for k, _ in pairs(encdec.paramx) do
    local b, _, d, dc1, dc2 = check_grad(encdec, k, p)
    if not b then return b, k, d, dc1, dc2 end
  end
  return true
end

function check_grad(encdec, k, p)
  print("Checking " .. k .. "...")
  local paramx = encdec.paramx[k]
  paramx.paramdx:zero()
  local x = paramx.paramx
  
  local prob = p or math.min(1, 100/x:size(1))

  local state = encdec:new_state("a b. cd.", "a b. cd")
  local length = state.enc.x:size(1)
  encdec.encoder:init_encoders(length)
  encdec.decoder:init_encoders(length-1)
  encdec:reset()
  encdec:reset_state(state)
  -- forward
  local loss1 = encdec:fp(state, length, length-1)
  encdec:reset()
  encdec:reset_state(state)
  local loss2 = encdec:fp(state, length, length-1)
  assert(loss1 == loss2, "Something is wrong in forward pass. Losses are not equal. " .. loss1 .. "  -  " .. loss2)
  assert(loss1 ~= 0, "Loss should not be zero.")
  -- backward
  encdec:bp(state, length, length-1)

  -- compute true gradient:
  local dC = paramx.paramdx

  -- compute numeric approximations to gradient:
  local eps = 1e-4
  for i = 1,dC:size(1) do
    if(torch.bernoulli(prob) == 1) then
      x[i] = x[i] + eps
      encdec:reset()
      encdec:reset_state(state)
      local C1 = encdec:fp(state,length,length-1)
      x[i] = x[i] - 2 * eps
      encdec:reset()
      encdec:reset_state(state)
      local C2 = encdec:fp(state,length,length-1)
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
    capacity = 2,
    dropout = 0,
    batch_size = 1,
    vocab = data.vocab_utf8,
    lookup=true,
    embedding_size=10,
  }
  local enc_layers = {
    [1] = { layer_type = "LSTMLayer", depth = 2 },
   -- [2] = { layer_type = "AttentionSkipLayer", skip = 2 }
  }
  local dec_layers = {
    [1] = { layer_type = "AttentionLSTMLayer", attention_capacity = 2, depth = 2  } ,
  }
  local encoder = AttentionEncoderDecoder(init_params, enc_layers, dec_layers)
  encoder.encoder:random_init(0.1)
  encoder.decoder:random_init(0.1)
  return check_all_grads(encoder)
end