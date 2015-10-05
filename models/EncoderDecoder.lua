-- use memory state when connecting different layers and output only for recurrence
local util = require('util')
require("models/StackEncoder")
require("models/StackDecoder")
require("pl")

local EncoderDecoder = torch.class('EncoderDecoder')

function EncoderDecoder:__init(params, enc_layers, dec_layers)
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

function EncoderDecoder:new_state(encode, decode, mapping, old_s)
  local s = old_s or {}
  s.map = mapping
  s.enc = self.encoder:new_state(encode,s.enc)
  s.dec = self.decoder:new_state(decode,s.dec)
  return s
end

function EncoderDecoder:reset_state(s)
  self.encoder:reset_state(s.enc)
  self.decoder:reset_state(s.dec)
end

function EncoderDecoder:reset()
  self.encoder:reset()
  self.decoder:reset()
end

function EncoderDecoder:reset_s()
  self.encoder:reset_s()
  self.decoder:reset_s()
end

function EncoderDecoder:run(state)
  self:disable_training()
  local perp = 0
  reset_state(state)
  while state.enc.pos < state.enc:size(1) do
    local p1 = self:fp(state,state.enc.x:size(1),state.dec.x:size(1))
    perp = perp + p1
  end
  self:enable_training()
  return perp
end

function EncoderDecoder:fp(state, enc_length, dec_length)
  self:reset_s()
  local off_enc = state.enc.pos
  local off_dec = state.dec.pos
  local mapping = state.map or {[enc_length]=0}
  self.encoder:fp(state.enc, enc_length, true)
  for i = off_enc+1,state.enc.pos do
    local d = mapping[i]
    if d then
      d = d - off_dec
      local e  = i - off_enc
      for l=1, #self.encoder.layers do
        local e_l = self.encoder.layers[l]
        local d_l = self.decoder.layers[l]
        if d > 0 then
          util.add_table(d_l.s[d], e_l.s[e]) 
        else 
          assert(d_l.start_s, "Tying to copy state from encoder to start state of decoder, which decoder doesn't have!")
          util.add_table(d_l.start_s, e_l.s[e])  
        end
      end
    end
  end

  return self.decoder:fp(state.dec, dec_length, true)
end

function EncoderDecoder:bp(state, enc_length, dec_length)
  self.decoder:bp(state.dec, dec_length)
  local off_enc = state.enc.pos-enc_length
  local off_dec = state.dec.pos-dec_length
  local mapping = state.map or {[enc_length]=0}
  for i = off_enc+1,state.enc.pos do
    local d = mapping[i]
    if d then
      d = d - off_dec
      local e  = i - off_enc
      for l=1, #self.encoder.layers do
        local e_l = self.encoder.layers[l]
        local d_l = self.decoder.layers[l]
        util.add_table(e_l.ds[e], d_l.ds[d])
      end
    end
  end
  self.encoder:bp(state.enc, enc_length)
end

function EncoderDecoder:setup(batch_size)
  self.encoder:setup(batch_size)
  self.decoder:setup(batch_size)
end

function EncoderDecoder:save(f)
  local t = {}
  t.encoder = self.encoder:params()
  t.decoder = self.decoder:params()
  torch.save(f,t)
end

function EncoderDecoder:load(f)
  local t = torch.load(f)
  self.encoder:set_params(t.encoder)
  self.decoder:set_params(t.decoder)
end

function EncoderDecoder:disable_training()
  self.encoder:disable_training()
  self.decoder:disable_training()
end

function EncoderDecoder:enable_training()
  self.encoder:enable_training()
  self.decoder:enable_training()
end
