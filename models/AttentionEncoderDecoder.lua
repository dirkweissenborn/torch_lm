require("models/EncoderDecoder")

local AttentionEncoderDecoder = torch.class('AttentionEncoderDecoder','EncoderDecoder')

function AttentionEncoderDecoder:new_state(encode, decode, old_s)
  local s = old_s or {}
  s.enc = self.encoder:new_state(encode,s.enc)
  s.dec = self.decoder:new_state(decode,s.dec)
  return s
end

function AttentionEncoderDecoder:fp(state, enc_length, dec_length)
  self.encoder:fp(state.enc, enc_length)
  local attention = {s={},ds={} }
  local attention_length = enc_length
  local last_layer = self.encoder.layers[#self.encoder.layers]
  if last_layer.skip then attention_length = math.ceil(enc_length/last_layer.skip) end
  for i = 1,attention_length do
    table.insert(attention.s, last_layer.out_s[i])
    table.insert(attention.ds, last_layer.out_ds[i])
  end
  state.dec.attention = attention
  return self.decoder:fp(state.dec, dec_length)
end

function AttentionEncoderDecoder:bp(state, enc_length, dec_length)
  self.decoder:bp(state.dec, dec_length)
  self.encoder:bp(state.enc, enc_length)
end

