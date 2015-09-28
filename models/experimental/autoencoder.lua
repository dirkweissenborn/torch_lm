-- use memory state when connecting different layers and output only for recurrence
local util = require('util')
require("models/StackEncoder")
require("models/StackDecoder")
require("pl")

local TextAutoEncoder = torch.class('TextAutoEncoder')

function TextAutoEncoder:__init(params, enc_layers, dec_layers)
  local dec_params = tablex.deepcopy(params)
  local enc_params = tablex.deepcopy(params)
  if enc_layers then enc_params.layers = enc_layers end
  if dec_layers then dec_params.layers = dec_layers end
  
  assert(enc_layers == nil or #enc_params.layers <= #dec_params.layers)
  self.encoder = StackEncoder(enc_params)
  self.decoder = StackDecoder(dec_params)

  self.paramx = {}
  if self.encoder.paramx then
    for k,v in pairs(self.encoder.paramx) do
      self.paramx["encoder-" .. k] = v
    end
    for k,v in pairs(self.decoder.paramx) do
      self.paramx["decoder-" .. k] = v
    end
  end
end

function TextAutoEncoder:new_state(encode, decode, mapping)
  local s = {}
  s.map = mapping
  s.enc = self.encoder:new_state(encode)
  s.dec = self.decoder:new_state(decode)
  return s
end

function TextAutoEncoder:reset_state(s)
  self.encoder:reset_state(s.enc)
  self.decoder:reset_state(s.dec)
end

function TextAutoEncoder:reset()
  self.encoder:reset()
  self.decoder:reset()
end

function TextAutoEncoder:reset_s()
  self.encoder:reset_s()
  self.decoder:reset_s()
end

function TextAutoEncoder:run(state)
  self:disable_training()
  local perp = 0
  reset_state(state)
  while state.enc.pos < state.enc:size(1) do
    local p1 = self:fp(state,100,100)
    perp = perp + p1
  end
  self:enable_training()
  return perp
end

function TextAutoEncoder:fp(state, enc_length, dec_length)
  self:reset_s()
  local off_enc = state.enc.pos
  local off_dec = state.dec.pos
  
  self.encoder:fp(state.enc, enc_length, true)
  for i = off_enc+1,state.enc.pos do
    local d = state.map[i]
    if d then
      d = d - off_dec
      local e  = i - off_enc
      for l=1, #self.encoder.layers do
        local e_l = self.encoder.layers[l]
        self.decoder.layers[l].out_s[d]:add(e_l.out_s[e])
      end
    end
  end

  return self.decoder:fp(state.dec, dec_length, true)
end

function TextAutoEncoder:bp(state, enc_length, dec_length)
  self.decoder:bp(state.dec, dec_length)
  local off_enc = state.enc.pos-enc_length
  local off_dec = state.dec.pos-dec_length
  for i = off_enc+1,state.enc.pos do
    local d = state.map[i]
    if d then
      d = d - off_dec
      local e  = i - off_enc
      for l=1, #self.encoder.layers do
        local e_l = self.encoder.layers[l]
        e_l.out_ds[e]:add(self.decoder.layers[l].out_ds[d])
      end
    end
  end
  self.encoder:bp(state.enc, enc_length)
end

function TextAutoEncoder:save(f)
  local t = {}
  t.encoder = self.encoder:params()
  t.decoder = self.decoder:params()
  torch.save(f,t)
end

function TextAutoEncoder:load(f)
  local t = torch.load(f)
  self.encoder:set_params(t.encoder)
  self.decoder:set_params(t.decoder)
end

function TextAutoEncoder:disable_training()
  self.encoder:disable_training()
  self.decoder:disable_training()
end

function TextAutoEncoder:enable_training()
  self.encoder:enable_training()
  self.decoder:enable_training()
end
