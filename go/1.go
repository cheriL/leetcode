package _go

import (
	"math"
	"sort"
	"strings"
)

func lengthOfLongestSubstring(s string) int {
	length := len(s)
	if length <= 0 {
		return length
	}

	subLen := int(1)

	for i := 0; i < length; i++ {
		rest := length - i
		if rest < subLen {
			break
		}

		for j := i + 1; j < length; j++ {
			subStr := s[i : j+1]
			if strings.Count(subStr, string(s[j])) > 1 {
				l := len(subStr) - 1
				if l > subLen {
					subLen = l
				}
				break
			}

			if j == length-1 {
				subLen = len(subStr)
				break
			}
		}
	}

	return subLen
}

func longestPalindrome(s string) string {
	length := len(s)
	if length <= 1 {
		return s
	} else if length == 2 {
		if s[0] == s[1] {
			return s
		} else {
			return string(s[0])
		}
	}

	var strList []string
	for i := 0; i < length; i++ {
		for j := i + 2; j <= length; j++ {
			if s[i] == s[j-1] {
				str := s[i:j]
				strList = append(strList, str)
			}
		}
	}

	var maxSubLen int
	var maxSubStr string
	for _, v := range strList {
		subLen := len(v)
		if subLen <= maxSubLen {
			continue
		}

		start, end := 0, subLen-1
		for {
			if start >= end {
				if maxSubLen < subLen {
					maxSubLen = subLen
					maxSubStr = v
				}
				break
			}

			if v[start] != v[end] {
				if maxSubLen < 1 {
					maxSubLen = 1
					maxSubStr = string(v[0])
				}
				break
			}

			start++
			end--
		}
	}

	return maxSubStr
}

// N字变换 1 <= numRows <= 1000
func convertN(s string, numRows int) string {
	length := len(s)
	if numRows == 1 || length <= numRows {
		return s
	}

	var newStr string

	indexInterval := numRows*2 - 2
	rowIndex := numRows

	for i := 0; i < numRows; i++ {
		if i == numRows-1 {
			rowIndex = numRows
		}
		interval := rowIndex*2 - 2

		for j := i; j < length; {
			newStr = newStr + string(s[j])
			j = j + interval
			if interval < indexInterval {
				interval = indexInterval - interval
			}
		}

		rowIndex--
	}

	return newStr
}

// 整数反转
func reverse(x int) int {
	var num int
	for x != 0 {
		num = num*10 + x%10
		x /= 10
	}

	if num > math.MaxInt32 || num < math.MinInt32 {
		return 0
	}

	return num
}

func myAtoi(s string) int {
	length := len(s)
	pos := int(1)
	result := int(0)
	index := 0
	for ; index < length && s[index] == ' '; index++ {
	}
	if index >= length {
		return 0
	}

	if s[index] == '+' {
		index++
	} else if s[index] == '-' {
		pos = -1
		index++
	}

	for ; index < length; index++ {
		if s[index] < 48 || s[index] > 57 {
			break
		}
		result = result*10 + int(s[index]-'0')
		if result*pos >= math.MaxInt32 {
			return math.MaxInt32
		}
		if result*pos <= math.MinInt32 {
			return math.MinInt32
		}
	}

	return result * pos
}

func threeSumClosest(nums []int, target int) int {
	result, maxAbs := int(0), float64(13001)

	sort.Ints(nums)
	for i := 0; i < len(nums)-2; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		j := i + 1
		k := len(nums) - 1
		for j < len(nums)-1 && j < k {
			if j > i+1 && nums[j] == nums[j-1] {
				j++
				continue
			}
			if k < len(nums)-1 && nums[k] == nums[k+1] {
				k--
				continue
			}

			sum := nums[i] + nums[j] + nums[k]
			abs := math.Abs(float64(sum - target))
			if abs == 0 {
				return sum
			}
			if abs < maxAbs {
				maxAbs = abs
				result = sum
			}
			if sum > target {
				k--
			} else {
				j++
			}
		}
	}

	return result
}

// 电话号码的字母组合
func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return []string{}
	}
	mapping := map[string][]string{
		"2": {"a", "b", "c"},
		"3": {"d", "e", "f"},
		"4": {"g", "h", "i"},
		"5": {"j", "k", "l"},
		"6": {"m", "n", "o"},
		"7": {"p", "q", "r", "s"},
		"8": {"t", "u", "v"},
		"9": {"w", "x", "y", "z"},
	}

	var comb func(start, end int) []string
	comb = func(start, end int) []string {
		if start == end {
			num := string(digits[start])
			return mapping[num]
		}

		var strList []string
		mid := (start + end) / 2
		leftStrList := comb(start, mid)
		rightStrList := comb(mid+1, end)
		for _, l := range leftStrList {
			for _, r := range rightStrList {
				strList = append(strList, l+r)
			}
		}
		return strList
	}

	return comb(0, len(digits)-1)
}

// 有效括号
func isValid(s string) bool {
	var stack []int32
	for _, v := range s {
		switch v {
		case '(', '[', '{':
			stack = append(stack, v)
		case ')':
			if length := len(stack); length > 0 {
				if stack[length-1] == '(' {
					stack = stack[:length-1]
				} else {
					return false
				}
			} else {
				return false
			}
		case ']':
			if length := len(stack); length > 0 {
				if stack[length-1] == '[' {
					stack = stack[:length-1]
				} else {
					return false
				}
			} else {
				return false
			}
		case '}':
			if length := len(stack); length > 0 {
				if stack[length-1] == '{' {
					stack = stack[:length-1]
				} else {
					return false
				}
			} else {
				return false
			}
		default:
			return false
		}
	}

	if len(stack) > 0 {
		return false
	}
	return true
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	//迭代哑节点
	dummyNode := &ListNode{}
	p := dummyNode
	for list1 != nil && list2 != nil {
		if list1.Val >= list2.Val {
			p.Next = list2
			list2 = list2.Next
		} else {
			p.Next = list1
			list1 = list1.Next
		}
		p = p.Next
	}
	if list1 == nil {
		p.Next = list2
	} else if list2 == nil {
		p.Next = list1
	}

	return dummyNode.Next
}

// 括号生成
func generateParenthesis(n int) []string {
	var res []string
	var dfs func(int, int, int, string)
	dfs = func(n, l, r int, path string) {
		if n == l && n == r {
			res = append(res, path)
			return
		}

		if l < n {
			dfs(n, l+1, r, path+"(")
		}
		if r < l {
			dfs(n, l, r+1, path+")")
		}
	}

	dfs(n, 0, 0, "")
	return res
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) <= 0 {
		return nil
	}
	//迭代哑节点
	mergeTwoList := func(list1 *ListNode, list2 *ListNode) *ListNode {
		dummyNode := &ListNode{}
		p := dummyNode
		for list1 != nil && list2 != nil {
			if list1.Val >= list2.Val {
				p.Next = list2
				list2 = list2.Next
			} else {
				p.Next = list1
				list1 = list1.Next
			}
			p = p.Next
		}
		if list1 == nil {
			p.Next = list2
		} else if list2 == nil {
			p.Next = list1
		}

		return dummyNode.Next
	}

	var binMerge func(start, end int) *ListNode
	binMerge = func(start, end int) *ListNode {
		if start == end {
			return lists[start]
		}
		mid := (start + end) / 2
		left := binMerge(start, mid)
		right := binMerge(mid+1, end)
		return mergeTwoList(left, right)
	}

	return binMerge(0, len(lists)-1)
}

/**两两交换链表中的节点
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func swapPairs(head *ListNode) *ListNode {
	dummyNode := &ListNode{
		Next: head,
	}
	p, q := dummyNode, head
	for q != nil && q.Next != nil {
		temp := q.Next
		q.Next = temp.Next
		temp.Next = q
		p.Next = temp
		p = q
		q = q.Next
	}
	return dummyNode.Next
}
