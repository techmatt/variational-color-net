
local ffi = require('ffi')
local Threads = require('threads')
Threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-------------------------------------------------------------------------------

local function makeThreadPool(opt)

    print('Initializing thread pool with ' .. opt.nDonkeys .. ' threads')
    local donkeys = nil
    do -- start K datathreads (donkeys)
        if opt.nDonkeys > 0 then
            local options = opt -- make an upvalue to serialize over to donkey threads
            donkeys = Threads(
                opt.nDonkeys,
                function()
                    -- TODO: load modules needed by donkeys here
                    require('image')
                end,
                function(idx)
                    opt = options -- pass to all donkeys via upvalue
                    tid = idx
                    local seed = opt.manualSeed + idx
                    torch.manualSeed(seed)
                    print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
                end
            )
        else -- single threaded data loading. useful for debugging
            donkeys = {}
            function donkeys:addjob(f1, f2) f2(f1()) end
            function donkeys:synchronize() end
        end
    end

    local nTest = 0
    donkeys:addjob(function() return 128 end, function(c) nTest = c end)
    donkeys:synchronize()
    assert(nTest > 0, "Failed to get nTest")
    print('Donkey test: ', nTest)

    return donkeys

end

return {
    makeThreadPool = makeThreadPool
}