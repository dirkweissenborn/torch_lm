local util = {}

function util.disable_training(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(util.disable_training)
    end
    return
  end
  if string.match(node.__typename, "Dropout") or 
      string.match(node.__typename, "GaussianNoise")  or
      string.match(node.__typename, "Flip") then
    node.train = false
  end
end

function util.enable_training(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(util.enable_training)
    end
    return
  end
  if string.match(node.__typename, "Dropout") or 
      string.match(node.__typename, "GaussianNoise") or
      string.match(node.__typename, "Flip") then
    node.train = true
  end
end

function util.cloneManyTimes_mt(net, T)
  require('torch')
  require('nngraph')
  require('fbcunn')
  require('gaussian_noise')

  --print("Cloning object " .. T .." times.")
  local threads = require 'threads'
  --threads.Threads.serialization('threads.sharedserialize')
  local clone = util.cloneManyTimes_st(net,1)[1]
  local nthread = 7
  local batch = 40
  local params, gradParams = net:parameters()
  local clone_params, clone_gradParams = clone:parameters()
  local dummy = torch.zeros(1):typeAs(params[1])
  for i = 1, #params do
    clone_params[i]:set(dummy:storage(),1,dummy:size())
    clone_gradParams[i]:set(dummy:storage(),1,dummy:size())
  end
  clone_params, clone_gradParams = clone:parameters()

  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(clone)
  local storage = mem:storage()
  local pool = threads.Threads(nthread,
    function(i)
      require('torch')
      require('nngraph')
      require('fbcunn')
      require('gaussian_noise')
    end ,
    function(i)
      t_params = clone_params
      t_gradParams = clone_gradParams
      t_storage = storage
      t_mem = torch.MemoryFile("w")
    end
  )

  local clones = {}
  local ct = 0
  for i=1,math.ceil(T/(batch*nthread)) do
    for j = 1, nthread do
      ct = ct+1
      local size = batch
      if ct >= math.ceil(T/batch) then size = T-(ct-1)*batch end

      if size > 0 then pool:addjob(
        function()
          local clones2 = {}
          for t = 1, size do
            -- We need to use a new reader for each clone.
            -- We don't want to use the pointers to already read objects.
            local reader = torch.MemoryFile(t_storage, "r"):binary()
            local clone = reader:readObject()
            reader:close()
            local cloneParams, cloneGradParams = clone:parameters()
            for i = 1, #t_params do
              cloneParams[i]:set(t_params[i])
              cloneGradParams[i]:set(t_gradParams[i])
            end
            collectgarbage()
            table.insert(clones2,clone)
          end
          t_mem:close()
          t_mem = torch.MemoryFile("w")
          t_mem:binary():writeObject(clones2)
          return t_mem:storage()
        end,
        function(clones2_storage)
          local reader = torch.MemoryFile(clones2_storage, "r"):binary()
          local clones2 = reader:readObject()
          reader:close()
          for _,clone in pairs(clones2) do
            local cloneParams, cloneGradParams = clone:parameters()
            for i = 1, #params do
              cloneParams[i]:set(params[i]:storage(),1,params[i]:size())
              cloneGradParams[i]:set(gradParams[i]:storage(),1,gradParams[i]:size())
            end
            table.insert(clones,clone)
            if(#clones % 100 == 0) then
              collectgarbage()
              --print(#clones)
            end
          end
        end)
      end
    end
    pool:synchronize()
  end

  pool:terminate()
  mem:close()
  collectgarbage()

  return clones
end

function util.cloneManyTimes(net, T)
  local clones = {}
  local params, gradParams = net:parameters()
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  local reader = torch.MemoryFile(mem:storage(), "r"):binary()

  local clone = reader:readObject()
  reader:close()
  mem:close()
  local clone_params, clone_gradParams = clone:parameters()
  if #params > 0 then 
    local dummy = torch.zeros(1):typeAs(params[1])
    for i = 1, #params do
      clone_params[i]:set(dummy)
      clone_gradParams[i]:set(dummy)
    end
  end
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(clone)

  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    clones[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end

function util.replace_table(to, from)
  assert(#to == #from)
  for k,v in pairs(from) do
    to[k]:resizeAs(v):copy(v)
  end
end

function util.add_table(to, from)
  assert(#to == #from)
  for k,v in pairs(from) do
    to[k]:add(v)
  end
end

function util.zero_table(t)
  for _,v in pairs(t) do
    v:zero()
  end
end

function util.f3(f)
  return string.format("%.3f", f)
end

function util.d(f)
  return string.format("%d", math.floor(f+0.5))
end

return util


