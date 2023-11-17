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

type MyString struct {
	str []rune
}

func NewMyString(s string) MyString {
	str := make([]rune, 0, len(s))
	for _, v := range s {
		str = append(str, v)
	}
	return MyString{str: str}
}
func (ms MyString) Len() int               { return len(ms.str) }
func (ms MyString) Less(i int, j int) bool { return ms.str[i] < ms.str[j] }
func (ms MyString) Swap(i int, j int)      { ms.str[i], ms.str[j] = ms.str[j], ms.str[i]; return }
func (ms MyString) Equal(t MyString) bool {
	if ms.Len() != t.Len() {
		return false
	}
	for i := 0; i < ms.Len(); i++ {
		if ms.str[i] != t.str[i] {
			return false
		}
	}
	return true
}

// 242. 有效的字母异位词
func isAnagram(s string, t string) bool {
	ms, mt := NewMyString(s), NewMyString(t)
	sort.Sort(ms)
	sort.Sort(mt)
	return ms.Equal(mt)
}

// 49. 字母异位词分组
// 输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
// 输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
func groupAnagrams(strs []string) [][]string {
	myStrList := make([]MyString, 0, len(strs))
	for i := 0; i < len(strs); i++ {
		str := NewMyString(strs[i])
		sort.Sort(str)
		myStrList = append(myStrList, str)
	}

	strMappings := map[int][]string{
		0: {strs[0]},
	}
	for i := 1; i < len(strs); i++ {
		find := false
		for index := range strMappings {
			if myStrList[i].Equal(myStrList[index]) {
				strMappings[index] = append(strMappings[index], strs[i])
				find = true
				break
			}
		}
		if !find {
			strMappings[i] = []string{strs[i]}
		}
	}

	var results [][]string
	for _, strList := range strMappings {
		results = append(results, strList)
	}

	return results
}

// 42. 接雨水
func trap(height []int) int {
	var stack []int
	push := func(val int) {
		stack = append(stack, val)
	}
	pop := func() {
		if len(stack) > 0 {
			stack = stack[:len(stack)-1]
		}
	}

	sum, left := 0, height[0]
	for i := 1; i < len(height); i++ {
		if height[i] <= height[i-1] {
			//递减入栈
			push(height[i])
		} else {
			if height[i] >= left {
				for len(stack) > 0 {
					sum += left - stack[len(stack)-1]
					pop()
				}
				left = height[i]
			} else {
				count := 0
				for len(stack) > 0 && stack[len(stack)-1] < height[i] {
					sum += height[i] - stack[len(stack)-1]
					pop()
					count++
				}
				for count > 0 {
					push(height[i])
					count--
				}
				push(height[i])
			}
		}
	}
	return sum
}

// 200. 岛屿数量
func numIslands(grid [][]byte) int {
	inArea := func(g [][]byte, r, c int) bool {
		if r >= 0 && r < len(g) && c >= 0 && c < len(g[0]) {
			return true
		}
		return false
	}
	var markArea func([][]byte, [][]int, int, int, int)
	markArea = func(g [][]byte, t [][]int, r, c int, flag int) {
		if !inArea(g, r, c) {
			return
		}
		if g[r][c] == '0' || t[r][c] == flag {
			return
		}

		if g[r][c] == '1' {
			t[r][c] = flag
		}
		markArea(g, t, r, c-1, flag)
		markArea(g, t, r-1, c, flag)
		markArea(g, t, r, c+1, flag)
		markArea(g, t, r+1, c, flag)
	}

	table := make([][]int, len(grid))
	for i := 0; i < len(grid); i++ {
		table[i] = make([]int, len(grid[0]))
	}

	count := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == '1' && table[i][j] == 0 {
				count++
				markArea(grid, table, i, j, count)
			}
		}
	}

	return count
}

// 637. 二叉树的层平均值
func averageOfLevels(root *TreeNode) (results []float64) {
	if root == nil {
		return
	}

	queue := []*TreeNode{root}
	length, sum := 1, 0
	curCount, count := 1, 0
	for i := 0; i < length; i++ {
		node := queue[i]
		if node.Left != nil {
			queue = append(queue, node.Left)
			count++
		}
		if node.Right != nil {
			queue = append(queue, node.Right)
			count++
		}
		sum += node.Val
		if i == length-1 {
			res := float64(sum) / float64(curCount)
			results = append(results, res)
			length += count
			curCount = count
			sum, count = 0, 0
		}
	}
	return
}

// 77. 组合
func combine(n int, k int) [][]int {
	var results [][]int
	var path []int

	var backTrace func(int, int, int, []int, *[][]int)
	backTrace = func(idx int, n int, length int, path []int, results *[][]int) {
		if len(path) == length {
			result := make([]int, len(path))
			copy(result, path)
			*results = append(*results, result)
			return
		}

		for i := idx; i < n+1; i++ {
			path = append(path, i)
			backTrace(i+1, n, length, path, results)
			path = path[:len(path)-1]
		}
	}

	backTrace(1, n, k, path, &results)

	return results
}
