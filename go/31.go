package _go

//31. 下一个排列
func nextPermutation(nums []int) {
	if len(nums) <= 1 {
		return
	}

	desc := true
	tempMax := 101
	for _, v := range nums {
		if tempMax >= v {
			tempMax = v
		} else {
			desc = false
		}
	}
	if desc {
		for i := 0; i < len(nums)/2; i++{
			nums[i], nums[len(nums)-1-i] = nums[len(nums)-1-i], nums[i]
		}
		return
	}

	getMaxIndexFunc := func(ns []int) int {
		maxIndex, maxVal := -1, -1
		for k, v := range ns {
			if v >= maxVal {
				maxVal = v
				maxIndex = k
			}
		}
		return maxIndex
	}

	var doFunc func(nums []int)
	doFunc = func(ns []int) {
		maxIndex := getMaxIndexFunc(ns)
		if len(ns) <= 1 {
			return
		}
		if maxIndex == 0 {
			doFunc(ns[1:])
		} else if maxIndex != len(ns) - 1 && len(ns) > 2 {
			if ns[maxIndex+1] > ns[maxIndex-1] {
				maxTemp := ns[maxIndex+1]
				index := maxIndex+1
				for i := maxIndex+1; i < len(ns); i++ {
					if ns[i] > ns[maxIndex-1] && ns[i] < maxTemp {
						maxTemp = ns[i]
						index = i
					}
				}
				ns[index], ns[maxIndex-1] = ns[maxIndex-1], ns[index]
				//sort.Ints(ns[maxIndex:])
				doFunc(ns[maxIndex:])
			} else {
				doFunc(ns[maxIndex-1:])
			}
		} else {
			ns[maxIndex], ns[maxIndex-1] = ns[maxIndex-1], ns[maxIndex]
		}
	}

	doFunc(nums)
}
