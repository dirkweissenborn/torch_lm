require("data")
require("util")

function beam_search(decoder, str, n, l)
  decoder:disable_training()
  if decoder.batch_size > 1 then decoder:setup(1) end
  decoder:reset()
  local state = decoder:new_state(str)
  for i=1, state.x:size(1) do
    decoder:fp(state,1)
  end
  
  local rev_vocab = {}
  for k,v in pairs(decoder.vocab) do rev_vocab[v] = k end
  
  local function copy_s()
    local s = {}
    for k,l in pairs(decoder.layers) do 
      if l.start_s then
        if type(l.start_s) == "table" then
          s[k] = tablex.map(function(t) return t:clone() end, l.start_s)
        else 
          s[k] = l.start_s:clone()
        end  
      end
    end  
    return s
  end
  local s = copy_s()  
  
  local beam = {}
  local p = decoder.layers[#decoder.layers].out_s[1]
  local y, ix = torch.sort(p[1],1,true)
  
  for i = 1, math.min(n, p[1]:size(1)) do
    if #beam < n then
      local b = {}
      b.is = {ix[i]}
      b.chars = {rev_vocab[ix[i]]}
      b.p = y[i]
      b.s = s
      table.insert(beam, b)
    end
  end
  
  local input = decoder:new_state(" ")

  for j = 2, l do
    local new_beam = {}
    for i= 1,  #beam do
      local b = beam[i]
      local last = b.is[#b.is]
      --reset state
      input.x[1] = last
      input.pos = 0
      for k,l in pairs(decoder.layers) do
        if l.start_s then 
          if type(l.start_s) == "table" then
            util.replace_table(l.start_s, b.s[k])
          else
            l.start_s:copy(b.s[k])
          end
        end
      end
      decoder:fp(input, 1)
      local s = copy_s()
      local y,ix = torch.sort(p[1],1,true)
      for ii = 1, math.min(n, p[1]:size(1)) do
        local b2 = tablex.deepcopy(b)
        b2.s = s
        table.insert(b2.is, ix[ii])
        table.insert(b2.chars, rev_vocab[ix[ii]])
        b2.p = b.p + y[ii]
        table.insert(new_beam, b2)
      end
    end
    local i = 1
    for k,v in tablex.sortv(new_beam, function(x,y) return x.p > y.p end) do 
      if i < n+1 then
        beam[i] = v
        i= i+1
      end  
    end
  end
  return beam
end

function next_word(decoder, str, n, max_l)
  local beam = beam_search(decoder, str, n or 10, max_l or 20)
  local next = table.concat(beam[1].chars)
  local split = stringx.split(next," ")
  return str .. split[1] .. " "
end

function beam_search_encdec(encdec, str, n, l)
  encdec:disable_training()
  if encdec.decoder.batch_size > 1 then encdec:setup(1) end
  encdec:reset()
  local state = encdec:new_state(str,"")
  encdec:fp(state, state.enc.x:size(1),1)
  local decoder = encdec.decoder

  local rev_vocab = {}
  for k,v in pairs(decoder.vocab) do rev_vocab[v] = k end

  local function copy_s()
    local s = {}
    for k,l in pairs(decoder.layers) do
      if l.start_s then
        if type(l.start_s) == "table" then
          s[k] = tablex.map(function(t) return t:clone() end, l.start_s)
        else
          s[k] = l.start_s:clone()
        end
      end
    end
    return s
  end
  local s = copy_s()

  local beam = {}
  local p = decoder.layers[#decoder.layers].out_s[1]
  local y, ix = torch.sort(p[1],1,true)

  for i = 1, math.min(n, p[1]:size(1)) do
    if #beam < n then
      local b = {}
      b.is = {ix[i]}
      b.chars = {rev_vocab[ix[i]]}
      b.p = y[i]
      b.s = s
      table.insert(beam, b)
    end
  end

  local input = decoder:new_state(" ")

  for j = 2, l do
    local new_beam = {}
    for i= 1,  #beam do
      local b = beam[i]
      local last = b.is[#b.is]
      --reset state
      input.x[1] = last
      input.pos = 0
      for k,l in pairs(decoder.layers) do
        if l.start_s then
          if type(l.start_s) == "table" then
            util.replace_table(l.start_s, b.s[k])
          else
            l.start_s:copy(b.s[k])
          end
        end
      end
      decoder:fp(input, 1)
      local s = copy_s()
      local y,ix = torch.sort(p[1],1,true)
      for ii = 1, math.min(n, p[1]:size(1)) do
        local b2 = tablex.deepcopy(b)
        b2.s = s
        table.insert(b2.is, ix[ii])
        table.insert(b2.chars, rev_vocab[ix[ii]])
        b2.p = b.p + y[ii]
        table.insert(new_beam, b2)
      end
    end
    local i = 1
    for k,v in tablex.sortv(new_beam, function(x,y) return x.p > y.p end) do
      if i < n+1 then
        beam[i] = v
        i= i+1
      end
    end
  end
  return beam
end