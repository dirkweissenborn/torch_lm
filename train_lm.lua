-- parts of it from https://github.com/karpathy/char-rnn/blob/master/train.lua

require('pl')
require('optim')
require('optimize')
require('models/StackDecoder')
local util = require('util')
local data = require('data')
local model_utils = require('model_utils')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-data_dir','','data dir including input.txt')
cmd:option('-out_dir','#no_output','output of log and models')
cmd:option('-capacity',246,'capacity of network')
cmd:option('-type','lstm','type of layers: lstm|gru|rnn')
cmd:option('-epochs',50,'training epochs')
cmd:option('-depth',1,'depth of network')
cmd:option('-scaling',1,'scaling of layers')
cmd:option('-dropout',0,'non-recurrent dropout')
cmd:option('-gpu',0,'gpu index, <=0 for cpu')
cmd:option('-lr',2e-3,'learning rate')
cmd:option('-lr_decay',0.5,'decay of lr')
cmd:option('-decay_threshold',0.001,'minimal validation loss improvement for using lr decay')
cmd:option('-batch_size',100,'batchsize')
cmd:option('-seq_length',100,'character sequence length of minibatches')
cmd:option('-checkpoint',500,'checkpoint after every n minibatches')
cmd:option('-overwrite',false, 'overwrite old model')
cmd:option('-noise_variance',0, 'gaussian noise variance after lookup table')
cmd:option('-flip_prob',0, 'probability of character flip')
cmd:option('-lookup',false, 'use lookup instead of one-hot')
cmd:option('-word_level',false,'train word level model')
cmd:option('-train_frac',0.9,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
cmd:option('-beta1', 0, 'first order momentum of ADAM (best 0 for character based modeling)')
cmd:option('-beta2', 0.999, 'something close to 1; usually >= 0.99')
cmd:option('-weight_decay', 0, 'decay of weights')
cmd:option('-embedding_size', 200, 'size of symbol embeddings')
cmd:option('-layer','lstm',"lstm|gf(gated-feedback)|dg(depth-gated)|grid")
cmd:option('-repeats', 1, 'repeat each symbol how many times.')
cmd:option('-linewise', false, 'train linewise, in contrast to training on continuously on input text.')
cmd:option('-max_length', -1, 'Maximum length of line, when training linewise.')
cmd:option('-randomize', false, 'randomize linewise input.')
cmd:option('-skip', 1, 'use selective skip layers')

cmd:text()

-- parse input opt
opt = cmd:parse(arg)

if opt.gpu > 0 then
  require("setup_cuda")
  init_gpu(opt.gpu)
else require("setup_cpu") end

local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac}

local batchloader
local loader
if opt.linewise then
  batchloader = require('SplitLMLinewiseMinibatchLoader')
  loader = batchloader.create(opt.data_dir, opt.batch_size, -1, split_sizes, opt.randomize, opt.word_level)
else
  batchloader = require('SplitLMMinibatchLoader')
  loader = batchloader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes, opt.word_level)
end
opt.vocab = loader.vocab_mapping
print("Vocab size: " .. tablex.size(loader.vocab_mapping))

local name = opt.type .. '-cap' .. opt.capacity
if opt.layer ~= "lstm" then name = opt.layer .. "_" .. name end

if opt.depth   ~=1 then name = name .. "-depth" .. opt.depth end
if opt.scaling ~=1 then name = name .. "-scale" .. opt.scaling end
if opt.dropout > 0 then name = name .. "-dr" .. opt.dropout end
if opt.skip ~=1 then name = name .. "-skip" .. opt.skip end

-- Setup Architecture
local ls = {}
if opt.layer == "gf" then
  if stringx.startswith(opt.type, "lstm") then ls[1] = { layer_type = "GatedFeedbackLSTMLayer", depth =opt.depth }
  else ls[1] = { layer_type = "GatedFeedbackRecurrentLayer", type=opt.type, depth =opt.depth } end
elseif opt.layer == "dg" then
  ls[1] = { layer_type = "DepthGatedLSTMLayer", depth = opt.depth }
elseif opt.layer == "grid" then
  ls[1] = { layer_type = "GridLSTMLayer", depth = opt.depth }
elseif opt.skip > 1 and opt.depth>1 then
  for i=1, math.min(opt.depth,2) do
    if stringx.startswith(opt.type, "lstm") then
      table.insert(ls, { layer_type = "LSTMLayer", depth = 1 })
    else
      table.insert(ls, { layer_type = "RecurrentLayer", type=opt.type , depth = 1})
    end
  end
  for i=3, opt.depth do
    if stringx.startswith(opt.type, "lstm") then
      table.insert(ls, { layer_type = "SelectiveSkipLSTMLayer", skip = opt.skip })
    else
      table.insert(ls, { layer_type = "SelectiveSkipRecurrentLayer", type=opt.type, skip = opt.skip})
    end
  end
else
  for i=1, opt.depth do
    if stringx.startswith(opt.type, "lstm") then
      table.insert(ls, { layer_type = "LSTMLayer" })
    else
      table.insert(ls, { layer_type = "RecurrentLayer", type=opt.type})
    end
  end
end
opt.layers = ls

----------  Model & Data Setup -----------------
print("Setup. Can take a while ... ")

local decoder = StackDecoder(opt)

-- setup training data
local epoch_size = loader.ntrain
local step = 0
print("Epoch size: " .. epoch_size)

-- load last checkpoint if overwrite is false and it exists --
local log
local function append_to_log(s) if log then log:write("\n" .. s) end end

local model_file, opt_state_file
local optim_state = {}

local num_params = 0
local time_offset = 0
if opt.out_dir ~= "#no_output" then
  model_file = path.join(opt.out_dir, name .. ".model")
  local log_file = path.join(opt.out_dir, name .. ".log")
  opt_state_file =  path.join(opt.out_dir, name .. ".opt_state")
  
  if not opt.overwrite and path.exists(model_file) and path.exists(log_file) and path.exists(opt_state_file) then
    opt.overwrite = false
  else opt.overwrite = true end

  if not opt.overwrite then
    -- load old model
    decoder:load(model_file)
    for k,v in pairs(decoder:params()) do if opt[k] and k ~= "layers" then opt[k] = v end end

    -- set current step to last step in old log
    local log_lines = stringx.split(file.read(log_file),"\n")
    while log_lines[#log_lines] == "" or stringx.startswith(log_lines[#log_lines],"Loss") do 
      log_lines[#log_lines] = nil
    end
    step = math.ceil(tonumber(stringx.split(log_lines[#log_lines],"\t")[2]) * epoch_size)
    time_offset = tonumber(stringx.split(log_lines[#log_lines],"\t")[1])

    print("Starting from epoch " .. util.f3(step/epoch_size) .. "...")

    --load optimizer state
    optim_state = torch.load(opt_state_file)
    loader.batch_ix[1] = optim_state.batch_ix or 0
    step = optim_state.step or 0
    time_offset = optim_state.time or 0
    log = io.open(log_file,"w")
    log:write(table.concat(log_lines,"\n"))
  else 
    log = io.open(log_file,"w")
    log:write("Character LM Training")
  end
else
  opt.overwrite = true
end

for _,v in pairs(decoder:networks()) do num_params = num_params + v:getParameters():size(1) end
opt.vocab = nil --leave out for printing

if opt.overwrite then
  optim_state.losses = {}
  optim_state.lengths = {}
  optim_state.learningRate = opt.lr
  -- ADAM
  optim_state.beta1 = opt.beta1
  optim_state.beta2 = opt.beta2

  append_to_log("options: " .. pretty.write(opt))
  append_to_log("Num params: " .. num_params)
  append_to_log("time\tepoch\tloss\tvalid_loss")
end

print(opt)
print("Num params: " .. num_params)
if log then log:flush() end

if use_cuda then cutorch.synchronize() end
collectgarbage()

---------- Code to run at checkpoint -------
local beginning_time = torch.tic() - time_offset
local total_cases = 0

function create_decoder_state(x,y,s)
  local _s = s or {}
  local max_len = loader.max_length
  if opt.max_length > 0 then max_len = math.min(max_len, opt.max_length) end
  
  if not _s.x or _s.x:size(2) ~= x:size(2) then
    _s.x = transfer_data(torch.Tensor(max_len, loader.batch_size))
    _s.y = transfer_data(torch.Tensor(max_len, loader.batch_size))
  end
  
  _s.len = math.min(x:size(1),max_len)
  _s.x:sub(1,_s.len,1,-1):copy(x:sub(1,_s.len,1,-1))
  _s.y:sub(1,_s.len,1,-1):copy(y:sub(1,_s.len,1,-1))
  _s.pos = 0
  return _s
end

local start_ss = {}
for k,l in pairs(decoder.layers) do
  if l.start_s then
    if type(l.start_s) == "table" then
      start_ss[k] = tablex.map(function(t) return t:clone() end, l.start_s)
    else start_ss[k] = l.start_s:clone() end
  end
end

local rev_vocab = {}
for k,v in pairs(decoder.vocab) do rev_vocab[v] = k end
local eval_state
local function run_split(split_index)
  -- save current start_s, and set start_s to zero
  if not opt.linewise then
    for k,l in pairs(decoder.layers) do
      if l.start_s then
        if type(l.start_s) == "table" then 
          util.replace_table(start_ss[k], l.start_s)
          util.zero_table(l.start_s)
        else 
          start_ss[k]:resizeAs(l.start_s):copy(l.start_s)
          l.start_s:zero()
        end
      end
    end
  end
  -- run
  decoder:disable_training()
  local loss = 0
  local words = 0
  local n = loader.split_sizes[split_index]
  
  loader:reset_batch_pointer(split_index)
  local total_len = 0
  for i = 1, n do
    local x, y = loader:next_batch(split_index)
    eval_state = create_decoder_state(x, y, eval_state)
    if opt.linewise then decoder:reset() end --if linewise training, then no carrying over of states
    local len = eval_state.len
    if opt.max_length > 0 then len = math.min(len, opt.max_length) end
    total_len = len + total_len
    local l = opt.seq_length
    for i=1,math.ceil(len/opt.seq_length) do -- if lines are longer than max seq_length, then run splits
      if i == math.ceil(len/opt.seq_length) then l = (len-1) % opt.seq_length + 1 end
      loss = loss + decoder:fp(eval_state, l)
    end
    if not opt.word_level then 
      for i = 0, len-1 do
        if rev_vocab[eval_state.x[eval_state.pos-i][1]]:match("%s") then
          words = words + 1
        end
      end
    else 
      words = words + len
    end
  end
  local word_loss = loss / words
  loss = loss / total_len

  -- reset to previous start_s
  if not opt.linewise then
    for k,l in pairs(decoder.layers) do
      if l.start_s then
        if type(l.start_s) == "table" then util.replace_table(l.start_s, start_ss[k])
        else l.start_s:copy(start_ss[k]) end
      end
    end
  end
  
  return loss, word_loss
end

optim_state.valid_loss = optim_state.valid_loss or 1e10

local function run_checkpoint()
  collectgarbage()
  if log then
    local loss = 0
    local train_loss = 0
    local total_train = 0
    for i = 1, #optim_state.losses do
      total_train = total_train + optim_state.lengths[i]
      train_loss = train_loss + optim_state.losses[i]
    end
    train_loss = train_loss / total_train

    if opt.val_frac > 0 then loss = run_split(2) else loss = train_loss end
    
    local epoch = step / epoch_size
    if optim_state.valid_loss > loss+1e-3 then -- only save this if loss got better
      optim_state.valid_loss = loss
      append_to_log(util.d(torch.toc(beginning_time)) .. "\t" ..
          util.f3(epoch) .. "\t" .. 
          util.f3(train_loss) .. "\t" .. util.f3(loss))

      decoder:save(model_file)
      optim_state.batch_ix = loader.batch_ix[1]
      optim_state.step = step
      optim_state.time = torch.toc(beginning_time)
      torch.save(opt_state_file, optim_state)
      log:flush()
      if opt.val_frac > 0 then print("Validation set perplexity : " .. util.f3(torch.exp(loss)))
      else print("Train loss (no validation split): " .. util.f3(torch.exp(loss)))  end
    elseif optim_state.step/opt.checkpoint < step/opt.checkpoint - 2 then
      print("Early stopping based on validation loss. No changes in past 3 checkpoints.")
      step = opt.epochs * epoch_size
    else
      --drop learning rate
      optim_state.learningRate = optim_state.learningRate * opt.lr_decay
      print("No validation set improvement! Decaying learning rate to: " .. optim_state.learningRate)
    end

  end
end

---------- Training Closure -------------
local params, grad_params = model_utils.combine_all_parameters(decoder:networks())

if opt.overwrite then params:uniform(-0.08, 0.08) end

-- init encoders and decoders
print("Creating decoders: " .. opt.seq_length .. "...")
decoder:init_encoders(opt.seq_length)
local last_loss = 0
local train_state
function feval(x)
  decoder:enable_training()

  if x ~= params then params:copy(x) end
  grad_params:zero()
  
  ------------------ get minibatch -------------------
  local x, y = loader:next_batch(1)
  train_state = create_decoder_state(x,y,train_state)

  -- forward
  if opt.linewise then decoder:reset() end --if linewise training, then no carrying over of states
  local len = train_state.len
  if opt.max_length > 0 then len = math.min(len, opt.max_length) end
  local loss = 0
  for i=1,math.ceil(len/opt.seq_length) do -- if lines are longer than max seq_length, then run splits
    local l = opt.seq_length
    if i == math.ceil(len/opt.seq_length) then l= (len-1) % opt.seq_length + 1 end
    loss = loss + decoder:fp(train_state, l)
    -- backward
    decoder:bp(train_state, l)
  end
  -- clip gradient element-wise
  grad_params:clamp(-5, 5)
  last_loss = loss
  return loss, grad_params
end

function feval2(x) 
  return last_loss, grad_params
end

---------- Training -----------------
print("Training.")
optim_state.valid_loss = optim_state.valid_loss or 10000
collectgarbage()
while step < (opt.epochs * epoch_size) do
  step = step + 1
  if opt.weight_decay > 0 then params:mul(1-opt.weight_decay) end
  local _, loss = optim.adam(feval, params, optim_state)
  local last_len = train_state.len
  local index = (step-1) % opt.checkpoint + 1
  optim_state.losses[index] = loss[1]
  optim_state.lengths[index] = last_len

  total_cases = total_cases + last_len * opt.batch_size
  
  local epoch = step / epoch_size
  if (step-1) % math.floor(epoch_size / 10 +0.5) == 0 or step % epoch_size == 0 then
    local norm_dw = grad_params:norm() / params:norm()
    local train_loss = 0
    local total_train = 0
    for i = 1, #optim_state.losses do
      total_train = total_train + optim_state.lengths[i]
      train_loss = train_loss + optim_state.losses[i]
    end
    train_loss = train_loss / total_train
    local wps = math.floor(total_cases / torch.toc(beginning_time))
    local since_beginning = util.d(torch.toc(beginning_time) / 60)
    print('epoch = ' .. util.f3(epoch) ..
        ', train perp. = ' .. util.f3(math.exp(train_loss)) ..
        ', wps = ' .. wps ..
        ', lr = ' .. string.format("%.4f", optim_state.learningRate) ..
        ', grad/params norm = ' .. util.f3(norm_dw) ..
        ', since beginning = ' .. since_beginning .. ' mins.')
  end

  if step % opt.checkpoint == 0 then run_checkpoint() end
end

run_checkpoint()

if test_frac > 0 then
  if log then decoder:load(model_file) end --load best model
  decoder:setup(1)
  if opt.linewise then
    loader = batchloader.create(opt.data_dir, 1,  opt.seq_length, split_sizes, opt.word_level, opt.randomize)
  else
    loader = batchloader.create(opt.data_dir, 1, opt.seq_length, split_sizes, opt.word_level)
  end
  local loss,loss_words = run_split(3)
  print("Test set perplexity : " .. util.f3(torch.exp(loss)))
  print("Test set perplexity (words) : " .. util.f3(torch.exp(loss_words)))
  append_to_log("Loss on test: " .. util.f3(loss))
  append_to_log("Loss on test (words): " .. util.f3(loss_words))
end

if log then io.close(log) end

return decoder
