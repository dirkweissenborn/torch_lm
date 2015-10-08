require('pl')
utf8 = require 'lua-utf8'

local data = {}
data.vocab_utf8  = {}
for i=1,256 do data.vocab_utf8[utf8.char(i)] = i end
data.vocab_utf8["<unk>"] = 257
data.vocab_utf8["<sos>"] = 259
data.vocab_utf8["<eos>"] = 258


function data.load(fname, vocab_map, words)
  local str = file.read(fname)
  if words then return data.convertWords(str,vocab_map)
  else return data.convertChars(str,vocab_map) end
end

function data.convertChars(str, vocab_map)
  local x = torch.zeros(#str+1)
  local vocab_idx = tablex.size(vocab_map)
  x[1] = vocab_map["<sos>"]
  for i = 1, #str do
    local char = str:sub(i,i)
    if vocab_map[char] == nil then
      vocab_idx = vocab_idx + 1
      vocab_map[char] = vocab_idx
    end
    x[i+1] = vocab_map[char]
  end
  return x, vocab_map
end

function data.convertWords(str, vocab_map)
  str = stringx.replace(str, '\n', '<eos>')
  str = stringx.split(str)
  local x = torch.zeros(#str)
  vocab_map = vocab_map or {}
  local vocab_idx = tablex.size(vocab_map)
  for i = 1, #str do
    if vocab_map[str[i]] == nil then
      vocab_idx = vocab_idx + 1
      vocab_map[str[i]] = vocab_idx
    end
    x[i] = vocab_map[str[i]]
  end
  return x, vocab_map
end

function data.replicate(x_inp, batch_size, x_old)
  local s = x_inp:size(1)
  local off = math.floor(s / batch_size)
  local x = x_old or transfer_data(torch.zeros(off, batch_size))
  for i = 1, batch_size do
    local start = (i - 1) * off + 1
    local finish = start + x:size(1) - 1
    x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
  end
  return x
end

function data.loadAnnotations(fname)
  local annotations = stringx.split(file.read(fname),"\n")
  local annos = {}
  local mapping = {}
  local length = 1
  for i = 1, #annotations-1 do
    local split      = stringx.split(annotations[i],"\t")
    table.insert(annos, split[2])
    mapping[tonumber(split[1])]  = length + 1
    for _,_ in utf8.codes(split[2]) do length = length + 1 end
  end
  return data.convertUTF8(table.concat(annos),true), mapping
end

return data