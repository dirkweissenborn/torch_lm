require('pl')
utf8 = require 'lua-utf8'

local data = {}

data.params = {
  num_chars = 257, --256+1 for unknown
  num_chars_extended = 259, -- +start and end decoding
  start_decoding = 259,
  end_decoding = 258
}

data.vocab_utf8  = {}
for i=1,256 do data.vocab_utf8[utf8.char(i)] = i end
data.vocab_utf8["<unk>"] = 257
data.vocab_utf8["<sd>"] = 259
data.vocab_utf8["<ed>"] = 258


function data.loadTextUTF8(fname, vocab_map)
  if path.exists(fname) then
    vocab_map  = vocab_map or {}
    if #vocab_map ~= 259 then
      for i=1,256 do vocab_map[utf8.char(i)] = i end
      vocab_map["<unk>"] = 257
      vocab_map["<sd>"] = 259
      vocab_map["<ed>"] = 258
    end
    local str = file.read(fname)
    return data.convertUTF8(str), vocab_map
  else 
    return nil 
  end  
end

function data.load(fname, vocab_map, words)
  local str = file.read(fname)
  if words then return data.convertWords(str,vocab_map)
  else return data.convertChars(str,vocab_map) end
end

function data.convertChars(str, vocab_map)
  local x = torch.zeros(#str)
  local vocab_idx = tablex.size(vocab_map)
  for i = 1, #str do
    local char = str:sub(i,i)
    if vocab_map[char] == nil then
      vocab_idx = vocab_idx + 1
      vocab_map[char] = vocab_idx
    end
    x[i] = vocab_map[char]
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

function data.convertUTF8(str, with_start)
  local length = 0
  for _,_ in utf8.codes(str) do length = length + 1 end
  --print(string.format("Loading... size of data = %d", length))
  if with_start then length = length + 2 end
  local x = torch.zeros(length)
  local i = 1
  if with_start then 
    i = 2
    x[1] = data_params.start_decoding
  end
  for _,c in utf8.codes(str) do
    if c > 256 then x[i] = 257
    else x[i] = c end
    i = i+1
  end
  if with_start then x[i] = data_params.end_decoding end
  return transfer_data(x)
end

function data.convertToUTF8(x)
  local res = {}
  for i=1,x:size(1) do
    table.insert(res,utf8.char(x[i]))
  end
  return table.concat(res)
end

function data.convertNumToUTF8(x)
  return utf8.char(x)
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