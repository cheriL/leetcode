package top150

import "strings"

func merge(nums1 []int, m int, nums2 []int, n int) {
	i, j, k := m, n, m+n
	for i > 0 && j > 0 {
		if nums1[i-1] > nums2[j-1] {
			nums1[k-1] = nums1[i-1]
			i--
		} else {
			nums1[k-1] = nums2[j-1]
			j--
		}
		k--
	}

	if i > 0 {
		for i > 0 {
			nums1[k-1] = nums1[i-1]
			i--
			k--
		}
	} else {
		for j > 0 {
			nums1[k-1] = nums2[j-1]
			j--
			k--
		}
	}
}

func removeElement(nums []int, val int) int {
	i, k := 0, 0
	for i < len(nums) {
		if nums[i] != val {
			nums[k] = nums[i]
			k++
		}
		i++
	}

	return k
}

func removeDuplicates(nums []int) int {
	i, k := 1, 1
	for i < len(nums) {
		if nums[i] != nums[i-1] {
			nums[k] = nums[i]
			k++
		}
		i++
	}
	return k
}

func removeDuplicates2(nums []int) int {
	if len(nums) <= 2 {
		return len(nums)
	}

	i, k := 2, 2
	for i < len(nums) {
		if nums[i] != nums[k-2] {
			nums[k] = nums[i]
			k++
		}
		i++
	}
	return k
}

func majorityElement(nums []int) int {
	count, result := 0, -1
	for _, num := range nums {
		if count == 0 {
			result = num
			count++
		} else {
			if num == result {
				count++
			} else {
				count--
			}
		}
	}

	return result
}

func maxProfit(prices []int) int {
	lowest, profit := prices[0], 0
	for i := 1; i < len(prices); i++ {
		if temp := prices[i] - lowest; temp > profit {
			profit = temp
		}
		if prices[i] < lowest {
			lowest = prices[i]
		}
	}
	return profit
}

func maxProfit2(prices []int) int {
	profits := make([]int, 0, len(prices))
	profits = append(profits, 0)

	for i := 1; i < len(prices); i++ {
		temp := profits[i-1]
		for j := 0; j < i; j++ {
			if profit := prices[i] - prices[j] + profits[j]; profit > temp {
				temp = profit
			}
		}
		profits = append(profits, temp)
	}

	return profits[len(profits)-1]
}

func canJump(nums []int) bool {
	dp := make([]byte, len(nums))
	dp[len(dp)-1] = 1

	for i := len(dp) - 1; i >= 0; i-- {
		for j := 0; j < i; j++ {
			if dp[i] == 1 && nums[j] >= i-j {
				dp[j] = 1
			}
		}
		if dp[0] == 1 {
			return true
		}
	}

	return false
}

func jump(nums []int) int {
	count, offset, maxPos := 0, 0, 0

	for i := 0; i < len(nums)-1; i++ {
		if temp := nums[i] + i; temp > maxPos {
			maxPos = temp
		}
		if i == offset {
			offset = maxPos
			count++
		}
	}

	return count
}

func canCompleteCircuit(gas []int, cost []int) int {
	startIdx, zeroIdx := -1, -1
	for i := 0; i < len(gas); i++ {
		if gas[i]-cost[i] == 0 {
			zeroIdx = i
		} else if gas[i]-cost[i] > 0 {
			startIdx = i
			break
		}
	}
	if startIdx == -1 {
		if zeroIdx >= 0 {
			startIdx = zeroIdx
		} else {
			return -1
		}
	}

	for i, j, count := startIdx, startIdx, 0; i < len(gas); {
		if startIdx == -1 {
			startIdx = i
		}

		if count = count + gas[j] - cost[j]; count < 0 {
			i++
			count, startIdx, j = 0, -1, i
			continue
		}

		if j < len(gas)-1 {
			j++
		} else {
			j = 0
		}

		if j == i {
			break
		}
	}

	return startIdx
}

func candy(ratings []int) int {
	stack := make([]int, len(ratings))
	k := 0
	push := func(val int) {
		stack[k] = val
		k++
	}
	pop := func() int {
		k--
		return stack[k]
	}

	lastRating, increaseNum, count := -1, 1, 0
	//弹栈并计数
	popAll := func(r int) {
		chgFlag := false
		if k > 0 {
			chgFlag = true
		}

		num := 1
		for last := r; k > 0; {
			val := pop()
			if val > last {
				num++
			} else if val == last {
				num = 1
			}

			if k != 0 {
				count += num
			} else {
				if num < increaseNum {
					count += increaseNum
				} else {
					count += num
				}
			}
			last = val
		}

		if chgFlag {
			increaseNum = 1
		}
	}

	for i := 0; i < len(ratings)-1; i++ {
		//递增
		if ratings[i] < ratings[i+1] {
			popAll(ratings[i])
			count += increaseNum
			increaseNum++
		} else {
			push(ratings[i])
		}

		lastRating = ratings[i]
	}

	//last rating
	if ratings[len(ratings)-1] <= lastRating {
		popAll(ratings[len(ratings)-1])
		count += 1
	} else {
		count += increaseNum
	}

	return count
}

func trap(height []int) int {
	if len(height) <= 2 {
		return 0
	}

	stack, k := make([]int, len(height)), 0
	push := func(val int) {
		stack[k] = val
		k++
	}
	pop := func() { k-- }
	length := func() int { return k }
	front := func() int {
		if k > 0 {
			return stack[k-1]
		}
		return -1
	}

	count := 0
	for i := 0; i < len(height)-1; i++ {
		if height[i] > height[i+1] {
			push(i)
		} else {
			baseline := height[i]
			for length() > 0 {
				idx := front()
				leftHeight := height[idx]
				if height[i+1] < leftHeight {
					count += (height[i+1] - baseline) * (i - idx)
					baseline = height[i+1]
					break
				} else {
					count += (leftHeight - baseline) * (i - idx)
					baseline = leftHeight
					pop()
				}
			}
		}
	}

	return count
}

func romanToInt(s string) int {
	romanMapping := map[int32]int{
		'I': 1,
		'V': 5,
		'X': 10,
		'L': 50,
		'C': 100,
		'D': 500,
		'M': 1000,
	}
	romanCounts := map[int32]int{}
	sum := 0
	for _, c := range s {
		romanCounts[c] += romanMapping[c]
		switch c {
		case 'V', 'X':
			romanCounts[c] -= romanCounts['I']
			romanCounts['I'] = 0
		case 'L', 'C':
			romanCounts[c] -= romanCounts['X']
			romanCounts['X'] = 0
		case 'D', 'M':
			romanCounts[c] -= romanCounts['C']
			romanCounts['C'] = 0
		default:
		}
	}

	for _, v := range romanCounts {
		sum += v
	}
	return sum
}

func intToRoman(num int) string {
	thousands := []string{"", "M", "MM", "MMM"}
	hundreds := []string{"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"}
	tens := []string{"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"}
	ones := []string{"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"}

	return thousands[num/1000] + hundreds[num%1000/100] + tens[num%100/10] + ones[num%10]
}

func lengthOfLastWord(s string) int {
	count := 0
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] != ' ' {
			count++
		} else {
			if count != 0 {
				break
			}
		}
	}
	return count
}

func longestCommonPrefix(strs []string) string {
	getCommonPrefix := func(str, str2 string) string {
		idx := 0
		for ; idx < len(str) && idx < len(str2); idx++ {
			if str[idx] != str2[idx] {
				break
			}
		}
		return str[:idx]
	}

	var binarySearch func(strs []string, start, end int) string
	binarySearch = func(strs []string, start, end int) string {
		if start == end {
			return getCommonPrefix(strs[start], strs[end])
		}

		mid := (start + end) / 2
		l := binarySearch(strs, start, mid)
		r := binarySearch(strs, mid+1, end)
		return getCommonPrefix(l, r)
	}

	return binarySearch(strs, 0, len(strs)-1)
}

func reverseWords(s string) string {
	strList := strings.Fields(s)
	for i := 0; i < len(strList)/2; i++ {
		strList[i], strList[len(strList)-i-1] = strList[len(strList)-i-1], strList[i]
	}
	return strings.Join(strList, " ")
}

func convert(s string, numRows int) string {
	var result string
	//2n-3+1
	factor := 1
	if numRows > 1 {
		factor = numRows*2 - 2
	}
	for i := 0; i < numRows; i++ {
		if i >= len(s) {
			break
		}
		result += s[i : i+1]

		var n int
		var roll bool
		if i == 0 || i == numRows-1 {
			n = factor
		} else {
			n = (numRows-i)*2 - 2
			roll = true
		}

		for j := i + n; j > 0 && j < len(s); {
			result += s[j : j+1]
			if roll {
				n = factor - n
			}
			j = j + n
		}
	}
	return result
}

func StrStr(haystack string, needle string) int {
	for i := 0; i < len(haystack); i++ {
		j := i
		for j < len(haystack) && j-i < len(needle) && haystack[j] == needle[j-i] {
			j++
			continue
		}
		if j-i == len(needle) {
			return i
		}
	}

	return -1
}

func fullJustify(words []string, maxWidth int) []string {
	return nil
}
