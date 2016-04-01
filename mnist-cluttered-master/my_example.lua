--[[
Copyright 2014 Google Inc. All Rights Reserved.

Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file or at
https://developers.google.com/open-source/licenses/bsd
]]

local mnist_cluttered = require 'mnist_cluttered'
require 'image'


local dataConfig = {megapatch_w=60, num_dist=4}
local dataInfo = mnist_cluttered.createData(dataConfig)

for epoch = 0,99 do

	-- print('Epoch '..epoch..'/'..20)

	-- train/test

	local observation, target = unpack(dataInfo.nextExample())
	-- print("observation size:", table.concat(observation:size():totable(), 'x'))
	-- print("targets:", target)

	-- print("Saving example.png")

	local formatted = image.toDisplayTensor({input=observation})
	image.save("clMNIST/example"..tostring(epoch)..".png", formatted)
	
	--local y = image.toDisplayTensor({input=target})
	--image.save("clMNIST/y"..tostring(epoch)..".png", target)
	local y = torch.totable(target)
	
	for i = 1,10 do
		if y[i] == 1 then
			local f = io.open("clMNIST/y"..tostring(epoch), "w")
			f:write(i - 1)
			f:close()
			break
		end
	end

end
