require("models/AttentionEncoderDecoder")
if not transfer_data then require("setup_cpu") end

function check_all_grads(encdec, p)
  local function check_encoder(enc)
    for i=#enc.layers,0,-1 do
      local l = enc.layers[i]
      if l then
        print("Checking " .. l.layer_type .. "-" .. i .. "...")
        for _,n in pairs(l:networks()) do
          if not  check_grad(encdec, n) then
            --return false
          end
        end  
      end  
    end
    return true
  end
  print("Checking Decoder")
  if(check_encoder(encdec.decoder)) then
    print("Checking Encoder")
    if(check_encoder(encdec.encoder)) then return true end
  end        
  return false
end

function check_grad(encdec, network)
  local x, dC = network:parameters()
  local x = x[1]:view(-1)
  local dC = dC[1]:view(-1)
  assert(x:norm(1) > 0, "Parameters should be initialized.")
  if not dC or dC:nElement() == 0 then return true end
  dC:zero()
  
  local prob = p or math.min(1, 100/x:size(1))
  local state = encdec:new_state("a", "a")
  local length = state.enc.x:size(1)
  encdec.encoder:init_encoders(length)
  encdec.decoder:init_encoders(length)
  encdec:reset()
  encdec:reset_state(state)
  -- forward
  local loss1 = encdec:fp(state, length, length)
  encdec:reset()
  encdec:reset_state(state)
  local loss2 = encdec:fp(state, length, length)

  assert(loss1 == loss2, "Something is wrong in forward pass. Losses are not equal. " .. loss1 .. "  -  " .. loss2)
  assert(loss1 ~= 0, "Loss should not be zero.")
  -- backward
  encdec:bp(state, length, length)  

  if dC:norm(1) == 0 then 
    print("Warn: Norm of gradient should not be zero.") 
    return false
  end
  local eps = 1e-4
  for i = 1,dC:size(1) do
    if(torch.bernoulli(prob) == 1 and dC[i] ~= 0) then
      x[i] = x[i] + eps
      encdec:reset()
      encdec:reset_state(state)
      local C1 = encdec:fp(state,length,length)
      x[i] = x[i] - 2 * eps
      encdec:reset()
      encdec:reset_state(state)
      local C2 = encdec:fp(state,length,length)
      x[i] = x[i] + eps
      local dC_est = (C1/ (2 * eps) - C2/ (2 * eps))
      local diff = math.abs(dC[i] - dC_est)
      if diff > 1.0e-6 then 
        print(string.format("Gradient: %.7f, Est. Gradient: %.7f, Diff: %.7f", dC[i], dC_est, diff))
        return false
      end
    end
  end
  return true
end

function check_example()
  local init_params = {
    capacity = 2,
    dropout = 0,
    batch_size = 1,
    vocab = {["a"]=1,["<sos>"]=2,["<eos>"]=3},
    lookup=true,
    embedding_size=10,
  }
  local enc_layers = {
    [1] = { layer_type = "LSTMLayer"},
    [2] = { layer_type = "RecurrentLayer" }
  }
  local dec_layers = {
    [1] = { layer_type = "LSTMLayer" } ,
    [2] = { layer_type = "RecurrentLayer" },
   -- [3] = { layer_type = "PredictionLayer" }
  }
  local encdec = EncoderDecoder(init_params, enc_layers, dec_layers)
  encdec.encoder:random_init(0.1)
  encdec.decoder:random_init(0.1)
  
  return check_all_grads(encdec)
end