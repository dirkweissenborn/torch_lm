-- Modified from https://github.com/oxford-cs-ml-2015/practical6 &
--  https://github.com/karpathy/char-rnn/blob/master/util/CharSplitLMLinewiseMinibatchLoader.lua
-- the modification included support for train/val/test splits
require("pl")
local tds = require 'tds'
local SplitLMLinewiseMinibatchLoader = {}
SplitLMLinewiseMinibatchLoader.__index = SplitLMLinewiseMinibatchLoader

function SplitLMLinewiseMinibatchLoader.create(data_dir, batch_size, max_length, split_fractions, random_batches, words)
  -- split_fractions is e.g. {0.9, 0.05, 0.05}

  local self = {}
  setmetatable(self, SplitLMLinewiseMinibatchLoader)

  local input_file = path.join(data_dir, 'input.txt')
  local vocab_file = path.join(data_dir, 'vocab_lw.t7')
  local tensor_file = path.join(data_dir, 'data_lw.t7')
  if words then
    vocab_file = path.join(data_dir, 'vocab_words_lw.t7')
    tensor_file = path.join(data_dir, 'data_words_lw.t7')
  end
  -- fetch file attributes to determine if we need to rerun preprocessing
  local run_prepro = false
  if not (path.exists(vocab_file) or path.exists(tensor_file)) then
    -- prepro files do not exist, generate them
    print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
    run_prepro = true
  else
    -- check if the input file was modified since last time we
    -- ran the prepro. if so, we have to rerun the preprocessing
    local input_attr = lfs.attributes(input_file)
    local vocab_attr = lfs.attributes(vocab_file)
    local tensor_attr = lfs.attributes(tensor_file)
    if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
      print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
      run_prepro = true
    end
  end
  if run_prepro then
    -- construct a tensor with all the data, and vocab file
    print('one-time setup: preprocessing input text file ' .. input_file .. '...')
    SplitLMLinewiseMinibatchLoader.text_to_tensor(input_file, vocab_file, tensor_file, words)
  end

  -- perform safety checks on split_fractions
  assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
  assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
  assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')

  print('loading data files...')
  self.vocab_mapping = torch.load(vocab_file)
  self.vocab_mapping["<sos>"] = self.vocab_mapping["<sos>"] or vocab_size+1
  self.vocab_mapping["<eos>"] = self.vocab_mapping["<eos>"] or vocab_size+2
  --local data = torch.load(tensor_file)
  local file = torch.DiskFile(tensor_file, 'r')
  file['binary'](file)
  file:referenced(false)
  local data = file:readObject()
  file:close()
  local len = #data

  local train_l = math.floor(len * split_fractions[1])
  local train = tds.hash()
  tablex.icopy(train, data, 1, 1, train_l)
  local valid_l = math.floor(len * split_fractions[2])
  local valid 
  if valid_l > 0 then
    valid = tds.hash()
    tablex.icopy(valid, data, 1, train_l+1, valid_l)
  end
  local test_l = math.floor(len * split_fractions[3])
  local test
  if test_l > 0 then
    test = tds.hash()
    tablex.icopy(test, data, 1, train_l+valid_l+1,test_l)
  end
  
  self.data = tds.hash()
  self.data[1] = train
  self.data[2] = valid
  self.data[3] = test

  -- count vocab
  self.vocab_size = 0
  for _ in pairs(self.vocab_mapping) do
    self.vocab_size = self.vocab_size + 1
  end

  -- self.batches is a table of tensors
  self.batch_size = batch_size
  self.max_length = 0
  print('Sorting by line-length...')
  local function sort_data(d)
    local lens = torch.IntTensor(#d)
    for i=1,#d do lens[i] = d[i]:size(1) end
    local sorted,sorted_ix = torch.sort(lens)
    self.max_length = math.max(sorted[sorted:size(1)]+1, self.max_length)
    if max_length > 0 then
      --cut sequences that are too long
      local j = sorted:size(1)
      while sorted[j] > max_length do
        j = j - 1
      end
      sorted_ix = sorted_ix:sub(1,j)
    end  
    lens = nil
    collectgarbage()
    
    return sorted_ix
  end

  self.length_sorted = {}
  local sorted_train = sort_data(train)
  self.length_sorted[1] = sorted_train
  self.ntrain = math.floor(sorted_train:size(1)/batch_size)

  self.nval = 0
  if valid then
    local sorted_valid = sort_data(valid)
    self.length_sorted[2] = sorted_valid
    self.nval = math.floor(sorted_valid:size(1)/batch_size)
  end
  self.ntest = 0
  if test then
    local sorted_test = sort_data(test)
    self.length_sorted[3] = sorted_test
    self.ntest = math.floor(sorted_test:size(1)/batch_size)
  end

  if max_length > 0 then -- there is a maximum length
    self.max_length = math.min(max_length, self.max_length)
  end
  
  self.split_sizes = { self.ntrain, self.nval, self.ntest }
  self.batch_ix = { 0, 0, 0 }
  self.perm = {}
  if random_batches then
    print('Batches randomized.')
    self.perm[1] = torch.randperm(self.ntrain)
    self.perm[2] = torch.randperm(math.max(1,self.nval))
    self.perm[3] = torch.randperm(math.max(1,self.ntest))
  else
    print('Batches not randomized.')
    self.perm[1] = torch.linspace(1,self.ntrain,self.ntrain)
    self.perm[2] = torch.linspace(1,math.max(1,self.nval),math.max(1,self.nval))
    self.perm[3] = torch.linspace(1,math.max(1,self.ntest),math.max(1,self.ntest))
  end

  print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d',
    self.ntrain, self.nval, self.ntest))
  collectgarbage()
  return self
end

function SplitLMLinewiseMinibatchLoader:reset_batch_pointer(split_index, batch_index)
  batch_index = batch_index or 0
  self.batch_ix[split_index] = batch_index
end

function SplitLMLinewiseMinibatchLoader:next_batch(split_index, split_symbol) 
  --split_symbol is used for parallel corpora such as in MT
  if self.split_sizes[split_index] == 0 then
    -- perform a check here to make sure the user isn't screwing something up
    local split_names = { 'train', 'val', 'test' }
    print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
    os.exit() -- crash violently
  end
  -- split_index is integer: 1 = train, 2 = val, 3 = test
  self.batch_ix[split_index] = self.batch_ix[split_index] + 1
  if self.batch_ix[split_index] > self.split_sizes[split_index] then
    self.batch_ix[split_index] = 1 -- cycle around to beginning
  end
  
  local eos = self.vocab_mapping["<eos>"]
  local sos = self.vocab_mapping["<sos>"]
  
  -- pull out the correct next batch
  local data = self.data[split_index]
  local i = self.perm[split_index][self.batch_ix[split_index]]-1
  local batch_size = self.batch_size
  local len1 = 0
  local len2 = 0
  local split_num
  if split_symbol then split_num = self.vocab_mapping[split_symbol] end
  for j=1,batch_size do
    local d = data[self.length_sorted[split_index][i*batch_size+j]]
    if split_symbol then
      local split = 1
      while split < d:size(1) and d[split] ~= split_num do split = split + 1 end
      len2 = math.max(len2, d:size(1)-split+1)
      len1 = math.max(len1, split)
    else
      len1 = math.max(len1, d:size(1)+1)
    end    
  end
  
  local xbatch = torch.Tensor(batch_size,len1):typeAs(data[1]):fill(eos)
  local ybatch = torch.Tensor(batch_size,len1):typeAs(data[1]):fill(eos)
  local xbatch2, ybatch2
  if split_symbol then
    if len2 == 0 then return self:next_batch(split_index, split_symbol) end
    xbatch2 = torch.Tensor(batch_size,len2):typeAs(data[1]):fill(eos)
    ybatch2 = torch.Tensor(batch_size,len2):typeAs(data[1]):fill(eos)
  end
  for j=1,batch_size do
    local d = data[self.length_sorted[split_index][i*batch_size+j]]
    if split_symbol then
      local split = 1
      while split < d:size(1) and d[split] ~= split_num do split = split + 1 end
      xbatch[j][1] = sos
      if split>1 then
        -- first splits should end at end of batch (e.g., in MT, source sentence should end where target begins)
        -- -> start at len1-split+1
        xbatch[j]:sub(len1-split+2, len1):copy(d:sub(1,split-1))
        xbatch[j]:sub(1,len1-split+1):fill(sos)
        ybatch[j]:sub(1, split-1):copy(d:sub(1,split-1))
      end
      xbatch2[j][1] = sos
      if split+1 <= d:size(1) then
        xbatch2[j]:sub(2, d:size(1)-split+1):copy(d:sub(split+1,d:size(1)))
        ybatch2[j]:sub(1, d:size(1)-split):copy(d:sub(split+1,d:size(1)))
      end
    else
      xbatch[j]:sub(2, d:size(1)+1):copy(d)
      xbatch[j][1] = sos
      ybatch[j]:sub(1, d:size(1)):copy(d)
    end
  end
  if split_symbol then
    return xbatch:t(), ybatch:t(), xbatch2:t(), ybatch2:t()
  else
    return xbatch:t(), ybatch:t()
  end
end

-- *** STATIC method ***
function SplitLMLinewiseMinibatchLoader.text_to_tensor(in_textfile, out_vocabfile, out_tensorfile, words)
  local timer = torch.Timer()

  print('loading text file...')
  local cache_len = 10000
  local tot_len = 0
  local f = io.open(in_textfile, "r")

  -- create vocabulary if it doesn't exist yet
  print('creating vocabulary mapping...')
  -- record all characters to a set
  local unordered = { ["<eos>"] = true, ["<sos>"] = true }
  local rawdata = f:read(cache_len)
  local word_split
  repeat
    if not words then
      for char in rawdata:gmatch '.' do
        if not unordered[char] then unordered[char] = true end
      end
      tot_len = tot_len + #rawdata
    else
      if word_split then rawdata = word_split[#word_split] .. rawdata end
      rawdata = stringx.replace(rawdata, '\n', ' <eos> ')
      rawdata = stringx.replace(rawdata, '\t', ' <split> ')
      word_split = stringx.split(rawdata,' ')
      for i=1, #word_split - 1 do -- do not add last word, because it might got split
      if word_split[i] ~= "" then
        if not unordered[word_split[i]] then unordered[word_split[i]] = true end
        tot_len = tot_len + 1
      end
      end
    end
    rawdata = f:read(cache_len)
  until not rawdata
  -- also add last word
  if words then
    tot_len = tot_len + 1
    if not unordered[word_split[#word_split]] then unordered[word_split[#word_split]] = true end
  end

  f:close()
  -- sort into a table (i.e. keys become 1..N)
  local ordered = {}
  for char in pairs(unordered) do ordered[#ordered + 1] = char end
  table.sort(ordered)
  -- invert `ordered` to create the char->int mapping
  local vocab_mapping = {}
  for i, char in ipairs(ordered) do
    vocab_mapping[char] = i
  end
  -- construct a tensor with all the data
  print('putting data into tensor...')
  
  local data = tds.hash()
  f = io.open(in_textfile, "r")
  for rawdata in f:lines() do
    local d     
    if not words then
      if #ordered < 256 then d = torch.ByteTensor(#rawdata)
      elseif #ordered < 32767 then d = torch.ShortTensor(#rawdata) end
      for i = 1, #rawdata do
        d[i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
      end
    else
      word_split = stringx.split(rawdata,' ')
      if #ordered < 256 then d = torch.ByteTensor(#word_split)
      elseif #ordered < 32767 then d = torch.ShortTensor(#word_split)
      else d = torch.IntTensor(#word_split) end

      for i=1, #word_split do
        if word_split[i] ~= "" then 
          d[i] = vocab_mapping[word_split[i]]
        end
      end
    end
    data[#data+1] = d
  end
  f:close()

  -- save output preprocessed files
  print('saving ' .. out_vocabfile)
  torch.save(out_vocabfile, vocab_mapping)
  print('saving ' .. out_tensorfile)
  local file = torch.DiskFile(out_tensorfile, 'w')
  file['binary'](file)
  file:referenced(false)
  file:writeObject(data)
  file:close()
end

return SplitLMLinewiseMinibatchLoader



