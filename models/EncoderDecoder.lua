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

  self.encoder = StackEncoder(enc_params)
  self.decoder = StackDecoder(dec_params)
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
  local loss = 0
  reset_state(state)
  while state.enc.pos < state.enc:size(1) do
    local l = self:fp(state,state.enc.x:size(1),state.dec.x:size(1))
    loss = loss + l
  end
  self:enable_training()
  return loss
end

function EncoderDecoder:fp(state, enc_length, dec_length)
  self:reset_s()
  --local off_enc = state.enc.pos
  --local off_dec = state.dec.pos
  --local mapping = state.map or {[enc_length]=0}
  self.encoder:fp(state.enc, enc_length, true)
  --[[for i = off_enc+1,state.enc.pos do
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
          assert(d_l.start_s, "Trying to copy state from encoder to start state of decoder, which decoder doesn't have!")
          util.add_table(d_l.start_s, e_l.s[e])  
        end
      end
    end
  end--]]
  local last_length = enc_length
  for l=1, #self.encoder.layers do
    local e_l = self.encoder.layers[l]
    local d_l = self.decoder.layers[l]
    assert(d_l.start_s, "Trying to copy state from encoder to start state of decoder, which decoder doesn't have!")
    last_length = e_l.last_length or last_length
    util.add_table(d_l.start_s, e_l.s[last_length])
  end

  return self.decoder:fp(state.dec, dec_length, true)
end

function EncoderDecoder:bp(state, enc_length, dec_length)
  self.decoder:bp(state.dec, dec_length)
  --[[local off_enc = state.enc.pos-enc_length
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
  end--]]
  local last_length = enc_length
  for l=1, #self.encoder.layers do
    local e_l = self.encoder.layers[l]
    local d_l = self.decoder.layers[l]
    last_length = e_l.last_length or last_length
    util.add_table(e_l.ds[last_length], d_l.ds[0])
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

function EncoderDecoder:networks(networks)
  networks = networks or {}
  self.encoder:networks(networks)
  self.decoder:networks(networks)
  return networks
end
