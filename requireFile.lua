
__requireFile__ = nil

if __requireFile__ == nil then

	local function makeRequireFile()
		local cache = {}
		return function(path)
			local fullpath = paths.concat(path)
			local rets = cache[fullpath]
			if rets == nil then
				rets = paths.dofile(path)
				cache[fullpath] = rets
			end
			return rets
		end
	end

	__requireFile__ = makeRequireFile()

end

return __requireFile__