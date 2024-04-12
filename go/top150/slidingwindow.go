package top150

import "math"

func minSubArrayLen(target int, nums []int) int {
	result := math.MaxInt32
	start, end, sum := 0, 0, 0
	for end < len(nums) && start <= end {
		sum += nums[end]
		if sum >= target {
			if result > end-start+1 {
				result = end - start + 1
			}
			sum -= nums[start]
			sum -= nums[end]
			start++
		} else if sum < target {
			end++
		}
	}

	if result == math.MaxInt32 {
		result = 0
	}

	return result
}
