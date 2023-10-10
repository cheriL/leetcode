//https://leetcode.cn/studyplan/top-interview-150/

package _go

// 88. 合并两个有序数组
func merge2(nums1 []int, m int, nums2 []int, n int) {
	i, j, k := m-1, n-1, m+n-1
	for i >= 0 && j >= 0 {
		if nums1[i] >= nums2[j] {
			nums1[k] = nums1[i]
			i--
		} else {
			nums1[k] = nums2[j]
			j--
		}
		k--
	}
	for j >= 0 {
		nums1[k] = nums2[j]
		j--
		k--
	}
}

// 27. 移除元素
func removeElement(nums []int, val int) int {
	length := len(nums)
	for i := 0; i < length; {
		if nums[i] == val {
			nums[i] = nums[length-1]
			length--
		} else {
			i++
		}
	}
	return length
}

// 26. 删除有序数组中的重复项
func removeDuplicates(nums []int) int {
	length := len(nums)
	if length <= 1 {
		return length
	}
	index := 1
	for i := 1; i < length; i++ {
		if nums[i] != nums[i-1] {
			nums[index] = nums[i]
			index++
		}
	}
	return index
}

// 121. 买卖股票的最佳时机
func maxProfit(prices []int) int {
	lowestPrice := prices[0]
	profit := 0
	for i := 1; i < len(prices); i++ {
		if prices[i]-lowestPrice > profit {
			profit = prices[i] - lowestPrice
		}
		if prices[i] < lowestPrice {
			lowestPrice = prices[i]
		}
	}
	return profit
}

// 122. 买卖股票的最佳时机 II 7,1,5,3,6,4
func maxProfit2(prices []int) int {
	dp := make([]int, len(prices))
	for i := 1; i < len(prices); i++ {
		maxP := 0
		for j := 0; j < i; j++ {
			profit := dp[j]
			if prices[i] > prices[j] {
				profit += prices[i] - prices[j]
			}
			if profit > maxP {
				maxP = profit
			}
		}
		dp[i] = maxP
	}
	return dp[len(prices)-1]
}
