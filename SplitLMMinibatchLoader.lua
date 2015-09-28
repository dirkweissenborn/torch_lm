-- Modified from https://github.com/oxford-cs-ml-2015/practical6 & 
--  https://github.com/karpathy/char-rnn/blob/master/util/CharSplitLMMinibatchLoader.lua
-- the modification included support for train/val/test splits

local SplitLMMinibatchLoader = {}
SplitLMMinibatchLoader.__index = SplitLMMinibatchLoader

function SplitLMMinibatchLoader.create(data_dir, batch_size, seq_length, split_fractions, words)
  -- split_fractions is e.g. {0.9, 0.05, 0.05}

  local self = {}
  setmetatable(self, SplitLMMinibatchLoader)

  local input_file = path.join(data_dir, 'input.txt')
  local vocab_file = path.join(data_dir, 'vocab.t7')
  local tensor_file = path.join(data_dir, 'data.t7')
  if words then
    vocab_file = path.join(data_dir, 'vocab_words.t7')
    tensor_file = path.join(data_dir, 'data_words.t7')
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
    SplitLMMinibatchLoader.text_to_tensor(input_file, vocab_file, tensor_file, words)
  end

  -- perform safety checks on split_fractions
  assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
  assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
  assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')

  print('loading data files...')
  self.vocab_mapping = torch.load(vocab_file)
  local data = transfer_data(torch.load(tensor_file))
  local len = data:size(1)
  
  local train_l = math.floor(len * split_fractions[1])
  local train = data:sub(1,train_l)
  local valid_l = math.floor(len * split_fractions[2])
  local valid = data:sub(train_l+1,train_l+valid_l)
  local test_l = math.floor(len * split_fractions[3])
  local test 
  if test_l > 0 then test = data:sub(train_l+valid_l+1,train_l+valid_l+test_l) end

  -- count vocab
  self.vocab_size = 0
  for _ in pairs(self.vocab_mapping) do
    self.vocab_size = self.vocab_size + 1
  end

  -- self.batches is a table of tensors
  self.batch_size = batch_size
  self.seq_length = seq_length
  
  print('reshaping tensors...')
  local function create_batches(d)
    -- cut off the end so that it divides evenly
    local d_len = d:size(1)
    if d_len % (batch_size * seq_length) ~= 0 then
      d = d:sub(1, batch_size * seq_length
          * math.floor(d_len / (batch_size * seq_length)))
    end

    local ydata = d:clone()
    ydata:sub(1, -2):copy(d:sub(2, -1))
    ydata[-1] = d[1]

    local xdata = d:view(batch_size, -1):t():split(seq_length, 1)
    ydata = ydata:view(batch_size, -1):t():split(seq_length, 1)
         
    return xdata, ydata
  end

  self.x_batches = {}
  self.y_batches = {}

  local x_batches_train, y_batches_train = create_batches(train)
  table.insert(self.x_batches, x_batches_train)
  table.insert(self.y_batches, y_batches_train)
  self.ntrain = #x_batches_train

  local x_batches_valid, y_batches_valid = create_batches(valid)
  table.insert(self.x_batches, x_batches_valid)
  table.insert(self.y_batches, y_batches_valid)
  self.nval = #x_batches_valid

  if test then
    local x_batches_test, y_batches_test = create_batches(test)
    table.insert(self.x_batches, x_batches_test)
    table.insert(self.y_batches, y_batches_test)
    self.ntest = #x_batches_test
  end

  self.split_sizes = { self.ntrain, self.nval, self.ntest }
  self.batch_ix = { 0, 0, 0 }

  print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', 
    self.ntrain, self.nval, self.ntest or 0))
  collectgarbage()
  return self
end

function SplitLMMinibatchLoader:reset_batch_pointer(split_index, batch_index)
  batch_index = batch_index or 0
  self.batch_ix[split_index] = batch_index
end

function SplitLMMinibatchLoader:next_batch(split_index)
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
  -- pull out the correct next batch
  local ix = self.batch_ix[split_index]
  return self.x_batches[split_index][ix], self.y_batches[split_index][ix]
end

-- *** STATIC method ***
function SplitLMMinibatchLoader.text_to_tensor(in_textfile, out_vocabfile, out_tensorfile, words)
  local timer = torch.Timer()

  print('loading text file...')
  local cache_len = 10000
  local tot_len = 0
  local f = io.open(in_textfile, "r")

  -- create vocabulary if it doesn't exist yet
  print('creating vocabulary mapping...')
  -- record all characters to a set
  local unordered = {}
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
      rawdata = stringx.replace(rawdata, '\n', '<eos>')
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
  local data 
  if not word_split then data = torch.ByteTensor(tot_len) -- store it into 1D first, then rearrange
  else data = torch.IntTensor(tot_len) end -- store it into 1D first, then rearrange
  f = io.open(in_textfile, "r")
  local currlen = 0
  rawdata = f:read(cache_len)
  word_split = nil
  repeat
    if not words then
      for i = 1, #rawdata do
        data[currlen + i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
      end
      currlen = currlen + #rawdata
    else
      if word_split then rawdata = word_split[#word_split] .. rawdata end
      rawdata = stringx.replace(rawdata, '\n', '<eos>')
      word_split = stringx.split(rawdata,' ')
      for i=1, #word_split - 1 do -- do not add last word, because it might got split
        if word_split[i] ~= "" then
          currlen = currlen + 1
       	  data[currlen] = vocab_mapping[word_split[i]]
        end
      end
    end
    rawdata = f:read(cache_len)
  until not rawdata
  -- also add last word
  if words then
    tot_len = tot_len + 1
    data[currlen+1] = vocab_mapping[word_split[#word_split]]
  end
  
  f:close()

  -- save output preprocessed files
  print('saving ' .. out_vocabfile)
  torch.save(out_vocabfile, vocab_mapping)
  print('saving ' .. out_tensorfile)
  torch.save(out_tensorfile, data)
end

return SplitLMMinibatchLoader

