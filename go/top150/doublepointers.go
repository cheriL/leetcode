package top150

import "sort"

func isPalindrome(s string) bool {
	isChar := func(c uint8) bool {
		if c >= '0' && c <= '9' ||
			c >= 'A' && c <= 'Z' ||
			c >= 'a' && c <= 'z' {
			return true
		}
		return false
	}

	for i, j := 0, len(s)-1; i < j && i < len(s) && j >= 0; {
		switch {
		case !isChar(s[i]):
			i++
			continue
		case !isChar(s[j]):
			j--
			continue
		}

		if s[i] != s[j] {
			if s[i] < '9' || s[j] < '9' {
				return false
			} else if s[i] != s[j]+32 && s[j] != s[i]+32 {
				return false
			}
		}

		i++
		j--
	}

	return true
}

func isSubsequence(s string, t string) bool {
	if len(s) == 0 {
		return true
	}

	for i, j := 0, 0; i < len(s) && j < len(t); j++ {
		if s[i] == t[j] {
			i++
		}

		if i == len(s) {
			return true
		}
	}

	return false
}

func twoSum(numbers []int, target int) []int {
	results := make([]int, 2)
	for i, j := 0, len(numbers)-1; i < j && i < len(numbers) && j >= 0; {
		if target == numbers[i]+numbers[j] {
			results[0], results[1] = i+1, j+1
			break
		} else if target < numbers[i]+numbers[j] {
			j--
		} else {
			i++
		}
	}

	return results
}

func maxArea(height []int) int {
	max := 0
	for i, j := 0, len(height)-1; i < j && i < len(height) && j >= 0; {
		h := height[i]
		if height[j] < h {
			h = height[j]
		}
		if temp := (j - i) * h; temp > max {
			max = temp
		}
		if height[i] < height[j] {
			i++
		} else {
			j--
		}
	}
	return max
}

func threeSum(nums []int) [][]int {
	var results [][]int
	sort.Ints(nums)
	for i := 0; i < len(nums)-2; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		for j, k := i+1, len(nums)-1; j < k && j < len(nums) && k > i; {
			if j > i+1 && nums[j] == nums[j-1] {
				j++
				continue
			}
			if 0 == nums[i]+nums[j]+nums[k] {
				results = append(results, []int{nums[i], nums[j], nums[k]})
				j++
				k--
			} else if 0 > nums[i]+nums[j]+nums[k] {
				j++
			} else {
				k--
			}
		}
	}
	return results
}
