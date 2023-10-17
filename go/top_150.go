//https://leetcode.cn/studyplan/top-interview-150/

package _go

import (
	"sort"
	"strings"
)

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

// 274. H 指数 3,0,6,1,5   1,3,1
func hIndex(citations []int) int {
	h := 0
	sort.Ints(citations)
	for i := len(citations) - 1; i >= 0; i-- {
		length := len(citations) - i
		count := 0
		for j := len(citations) - 1; j >= i; j-- {
			if citations[j] >= length {
				count++
			}
		}
		if count > h {
			h = count
		}
	}
	return h
}

// 134. 加油站
func canCompleteCircuit(gas []int, cost []int) int {
	rest := make([]int, len(gas))
	for i := 0; i < len(rest); i++ {
		rest[i] = gas[i] - cost[i]
	}

	start, left := 0, 0
	for start < len(rest) {
		if rest[start] > 0 {
			break
		}
		left += rest[start]
		start++
	}

	if left == 0 && start == len(rest) {
		return start - 1
	}

	for idx := start; idx < len(rest); idx++ {
		i, sum := idx, 0
		for i < len(rest) && sum >= 0 {
			sum += rest[i]
			i++
		}
		if i == len(rest) {
			if sum += left; sum >= 0 {
				return idx
			}
		}
		left += rest[idx]
	}

	return -1
}

// 58. 最后一个单词的长度
func lengthOfLastWord(s string) int {
	words := strings.Split(s, " ")
	for i := len(words) - 1; i >= 0; i-- {
		if length := len(words[i]); length > 0 {
			return length
		}
	}
	return 0
}

// 14. 最长公共前缀
func longestCommonPrefix(strs []string) string {
	commonPrefix := func(s1 string, s2 string) string {
		i := 0
		for i < len(s1) && i < len(s2) {
			if s1[i] != s2[i] {
				break
			}
			i++
		}
		return s1[:i]
	}

	var binarySearch func([]string, int, int) string
	binarySearch = func(strs []string, start, end int) string {
		if start == end {
			return strs[start]
		}
		mid := (start + end) / 2
		left := binarySearch(strs, start, mid)
		right := binarySearch(strs, mid+1, end)
		return commonPrefix(left, right)
	}
	return binarySearch(strs, 0, len(strs)-1)
}

// 135. 分发糖果
func candy(ratings []int) int {
	nums := make([]int, len(ratings))
	patch := func(idx int) {
		nums[idx] = 1
		l, r := idx-1, idx+1
		for i := l; i >= 0; i-- {
			if ratings[i] == ratings[i+1] {
				if nums[i] == 0 {
					nums[i] = 1
				}
			} else if ratings[i] > ratings[i+1] {
				if nums[i] <= nums[i+1] {
					nums[i] = nums[i+1] + 1
				}
			} else {
				break
			}
		}
		for i := r; i < len(ratings); i++ {
			if ratings[i] == ratings[i-1] {
				nums[i] = 1
			} else if ratings[i] > ratings[i-1] {
				nums[i] = nums[i-1] + 1
			} else {
				break
			}
		}
	}
	for i := 0; i < len(ratings); i++ {
		//寻找低点
		l, r := i-1, i+1
		if l >= 0 && ratings[l] < ratings[i] {
			continue
		}
		if r < len(ratings) && ratings[r] <= ratings[i] {
			continue
		}
		if nums[i] > 0 {
			continue
		}
		patch(i)
	}

	sum := 0
	for i := 0; i < len(nums); i++ {
		sum += nums[i]
	}
	return sum
}

// 151. 反转字符串中的单词
func reverseWords(s string) string {
	strList := strings.Fields(s)
	for i := 0; i < len(strList)/2; i++ {
		strList[i], strList[len(strList)-i] = strList[len(strList)-i], strList[i]
	}
	return strings.Join(strList, " ")
}

// 68. 文本左右对齐
func fullJustify(words []string, maxWidth int) []string {
	var results []string
	var subList []string
	var subLen int
	for i := 0; i < len(words); {
		if maxWidth >= subLen+len(subList)+len(words[i]) {
			subList = append(subList, words[i])
			subLen += len(words[i])
			i++
		} else {
			for j := 0; subLen < maxWidth; {
				subList[j] += " "
				subLen++
				if j >= len(subList)-2 {
					j = 0
				} else {
					j++
				}
			}
			results = append(results, strings.Join(subList, ""))
			subList, subLen = []string{}, 0
		}
	}
	last := strings.Join(subList, " ")
	results = append(results, last+strings.Repeat(" ", maxWidth-len(last)))
	return results
}

// 125. 验证回文串
func isPalindrome2(s string) bool {
	validate := func(c uint8) bool {
		if c > 47 && c < 58 ||
			c > 64 && c < 91 ||
			c > 96 && c < 123 {
			return true
		}
		return false
	}
	for i, j := 0, len(s)-1; i < j; {
		for !validate(s[i]) && i < j {
			i++
		}
		for !validate(s[j]) && j > i {
			j--
		}
		if i < j {
			if s[i] == s[j] {
			} else if s[i] > 64 && s[j] > 64 && (s[i] == s[j]+32 || s[j] == s[i]+32) {
			} else {
				return false
			}
		}
		i++
		j--
	}
	return true
}

// 392. 判断子序列
func isSubsequence(s string, t string) bool {
	i, j := 0, 0
	for ; i < len(s) && j < len(t); j++ {
		if s[i] == t[j] {
			i++
		}
	}
	if i < len(s) {
		return false
	}
	return true
}
