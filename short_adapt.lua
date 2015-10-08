require("models/EncoderDecoder")
local model_utils = require("model_utils")

function adapt_autoencoder(encdec, str, lr, its, params, grad_params, st_params)
  -- prepare params
  if not params then
    params, grad_params =
      model_utils.combine_all_parameters(unpack(tablex.map(function(l) return l.core_encoder end, encdec.encoder.layers)),
        unpack(tablex.map(function(l) return l.core_encoder end, encdec.decoder.layers)))
    st_params = params:clone()
  end

  st_params:zero()
  
  -- adapt  
  its = its or 1
  lr  = lr or 0.01
  encdec:disable_training()
  if encdec.decoder.batch_size > 1 then encdec:setup(1) end
  local state = encdec:new_state(str,str)
  
  for i = 1, its do
    grad_params:zero()
    encdec:reset_state(state)
    encdec:reset()
    encdec:fp(state, state.enc.x:size(1),state.dec.x:size(1))
    encdec:bp(state, state.enc.x:size(1),state.dec.x:size(1))
    st_params:add(-lr, grad_params)
    params:add(-lr, grad_params)
  end  
    
  return params, grad_params, st_params
end