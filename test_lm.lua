require('pl')
require('optim')
require('models/StackDecoder')
local SplitLMMinibatchLoader = require('SplitLMMinibatchLoader')
local util = require('util')
local model_utils = require('model_utils')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-data_dir','','data dir including input.txt')
cmd:option('-model_file','','path to models')
cmd:option('-gpu',0,'gpu index, <=0 for cpu')
cmd:option('-batch_size',1,'batchsize')
cmd:option('-seq_length',100,'character sequence length of minibatches')
cmd:option('-word_level',false,'train word level model')
cmd:option('-test_frac',0.05,'fraction of data that goes into train set')
cmd:option('-stp_lr', 1e-2, 'learning rate for short term parameters.')
cmd:option('-stp_decay', 0.5, 'weight decay of short term parameters.')
cmd:option('-optim', 'sgd', 'sgd|adam')
cmd:option('-beta2', 0.999, 'beta 2 if adam')
cmd:text()

-- parse input opt
opt = cmd:parse(arg)

if opt.gpu > 0 then
  require("setup_cuda")
  init_gpu(opt.gpu)
else require("setup_cpu") end

local decoder = StackDecoder(opt)
decoder:load(opt.model_file)

local loader = SplitLMMinibatchLoader.create(opt.data_dir,
  opt.batch_size, opt.seq_length, {(1-opt.test_frac)/2, (1-opt.test_frac)/2, opt.test_frac}, opt.word_level)

--ST params
local params, grad_params = model_utils.combine_all_parameters(decoder:networks())
local st_params, lt_params, optim_state
if opt.stp_lr > 0 then
  st_params  =params:clone():zero()
  lt_params = params:clone()
  optim_state = {}
  optim_state.learningRate = opt.stp_lr
  --ADAM
  if opt.optim == "adam" then
    optim_state.beta2 = opt.beta2
  end
end

local rev_vocab_loader = {}
for k,v in pairs(loader.vocab_mapping) do rev_vocab_loader[v] = k end

local rev_vocab = {}
for k,v in pairs(decoder.vocab) do rev_vocab[v] = k end

-- run
decoder:disable_training()
local loss = 0
local words = 0
local n = loader.split_sizes[3]

if opt.word_level then words = (n * opt.seq_length) end

local function run(eval_state)
  local l = decoder:fp(eval_state, opt.seq_length)
  if not opt.word_level then
    for i = 0, opt.seq_length-1 do
      if rev_vocab[eval_state.x[eval_state.pos-i][1]]:match("%s") then
        words = words + 1
      end
    end
  end
  return l
end

local eval_state = {}

local function transform(x)
  for i=1, x:size(1) do
    for j=1, x:size(2) do
      x[i][j] = decoder.vocab[rev_vocab_loader[x[i][j]]]
    end
  end
  return x
end

local optimizer = optim.sgd
if opt.optim == "adam" then
  optimizer = optim.adam
end

local eval_state
function create_decoder_state(x,y,s)
  local _s = s or {}
  if not _s.x or _s.x:size(1) ~= x:size(1) then
    _s.x = transfer_data(x:clone())
    _s.y = transfer_data(y:clone())
  end
  _s.x:copy(x)
  _s.y:copy(y)
  _s.pos = 0
  return _s
end

for i = 1, n do
  local x, y = loader:next_batch(3)
  x = transform(x)
  y = transform(y)
  eval_state = create_decoder_state(x,y,eval_state)
  local l = run(eval_state)
  if opt.stp_lr > 0 then
    optim.sgd( function(params)
      grad_params:zero()

      decoder:bp(eval_state, opt.seq_length)
      grad_params:clamp(-5, 5)
      st_params:mul(1-opt.stp_decay)
      return l, grad_params
    end, st_params, optim_state)
    params:copy(lt_params):add(st_params)
  end
  loss = loss + l
end

local word_loss = loss / words
loss = loss / (n * opt.seq_length)

print("Loss: " .. util.f3(loss))
print("Loss (words): " .. util.f3(word_loss))