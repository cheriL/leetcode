//https://leetcode.cn/tag/array/problemset/
//https://leetcode.cn/tag/dynamic-programming/problemset/

package _go

import (
	"fmt"
	"math"
	"sort"
	"strconv"
)

// 4. 寻找两个正序数组的中位数
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	length := len(nums1) + len(nums2)
	if length == 0 {
		return float64(0)
	}
	size := length/2 + 1
	data := make([]int, 0, size)
	for i, j := 0, 0; len(data) < size; {
		if len(nums1) > i && len(nums2) > j {
			if nums1[i] <= nums2[j] {
				data = append(data, nums1[i])
				i++
			} else {
				data = append(data, nums2[j])
				j++
			}
		} else if len(nums1) <= i {
			data = append(data, nums2[j])
			j++
		} else {
			//len(nums2) <= j
			data = append(data, nums1[i])
			i++
		}
	}

	if val := length % 2; val > 0 {
		return float64(data[length/2])
	} else {
		return (float64(data[length/2]) + float64(data[length/2-1])) / 2
	}
}

// 18. 四数之和
func fourSum(nums []int, target int) [][]int {
	// n数之和通用解法,排序+固定枚举前n-2个数字，双指针搜索最后两个数字。
	// 每个数字向后搜索时直接去重

	//排序
	merge := func(s, m, e int, nums []int) {
		sNums := make([]int, e-s)
		i, j, k := s, m, 0
		for {
			if i < m && j < e {
				if nums[i] <= nums[j] {
					sNums[k] = nums[i]
					i++
				} else {
					sNums[k] = nums[j]
					j++
				}
			} else if i >= m && j < e {
				sNums[k] = nums[j]
				j++
			} else if i < m && j >= e {
				sNums[k] = nums[i]
				i++
			} else {
				break
			}
			k++
		}
		for i := s; i < e; i++ {
			nums[i] = sNums[i-s]
		}
	}
	var mergeSort func(int, int, []int)
	mergeSort = func(start, end int, nums []int) {
		if start == end || end == start+1 {
			return
		}
		mid := (start + end) / 2
		mergeSort(start, mid, nums)
		mergeSort(mid, end, nums)
		merge(start, mid, end, nums)
	}

	mergeSort(0, len(nums), nums)

	var results [][]int
	for i := 0; i < len(nums)-3; i++ {
		//去重
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		for j := i + 1; j < len(nums)-2; j++ {
			//去重
			if j > i+1 && nums[j] == nums[j-1] {
				continue
			}
			//双指针
			p, q := j+1, len(nums)-1
			for p < q {
				if sum := nums[i] + nums[j] + nums[p] + nums[q]; sum == target {
					results = append(results, []int{nums[i], nums[j], nums[p], nums[q]})
					p++
					q--
					for p < q && nums[p] == nums[p-1] {
						p++
					}
					for p < q && nums[q] == nums[q+1] {
						q--
					}
				} else if sum > target {
					q--
				} else {
					p++
				}
			}
		}
	}
	return results
}

// 26. 删除有序数组中的重复项
func removeDuplicates(nums []int) int {
	index := 1
	for i := 1; i < len(nums); {
		if nums[i] != nums[i-1] {
			nums[index] = nums[i]
			index++
		}
		i++
	}
	return index
}

// 31. 下一个排列
func nextPermutation(nums []int) {
	if len(nums) <= 1 {
		return
	}
	//指定范围内寻找最大值
	findMaxIndex := func(nums []int, start, end int) int {
		index, max := -1, -1
		for i := start; i < end && i < len(nums); i++ {
			if nums[i] > max {
				max = nums[i]
				index = i
			}
		}
		return index
	}

	//判断是否降序; 及获取指定范围内比 target 大的值中最小的值（刚好大于target）的index
	isDesc := func(nums []int, start, end int, target int) (bool, int) {
		flag := true
		curr := nums[start]
		just, justIndex := 101, -1
		for i := start; i < end && i < len(nums); i++ {
			if nums[i] > curr {
				flag = false
			}
			curr = nums[i]
			if nums[i] > target && nums[i] < just {
				just = nums[i]
				justIndex = i
			}
		}
		return flag, justIndex
	}

	//在[start,end)中找最大值的index:
	//	*index==end-1:直接交换末尾两个值；
	//   *index==start:判断范围内是否为降序，如果是，则直接生序排序；否则start=start+1，循环执行；
	//	*index==x:判断[x,end)范围内是否降序，并获取该范围内刚好比nums[index-1]大的值的justIndex:
	//		*如果降序，且存在justIndex，将justIndex与index-1的值交换，对[index,end)做升序排序；
	//       *如果降序，但不存在justIndex，将index与index-1的值交换，对[index,end)做升序排序；
	//       *如果非降序，start=index+1,循环执行；
	index, flag := -1, false
	for start, end := 0, len(nums); !flag; {
		index = findMaxIndex(nums, start, end)
		switch index {
		case end - 1:
			nums[index], nums[index-1] = nums[index-1], nums[index]
			flag = true
		case start:
			desc, _ := isDesc(nums, start, end, -1)
			if desc {
				sort.Ints(nums)
				flag = true
			} else {
				start += 1
				index = findMaxIndex(nums, start, end)
			}
		default:
			leftVal := nums[index-1]
			desc, justIndex := isDesc(nums, index+1, end, leftVal)
			if desc {
				if justIndex > 0 {
					nums[justIndex], nums[index-1] = nums[index-1], nums[justIndex]
					sort.Ints(nums[index:])
					flag = true
				} else {
					nums[index], nums[index-1] = nums[index-1], nums[index]
					sort.Ints(nums[index:])
					flag = true
				}
			} else {
				start = index + 1
				index = findMaxIndex(nums, start, end)
			}
		}
	}
}

// 36. 有效的数独
func isValidSudoku(board [][]byte) bool {
	var rowSet [9][9]byte
	var colSet [9][9]byte
	var boxSet [9][9]byte

	for i, row := range board {
		for k, v := range row {
			if v == '.' {
				continue
			}
			val := v - '1'
			rowSet[i][val] += 1
			colSet[k][val] += 1
			boxIndex := i/3*3 + k/3
			boxSet[boxIndex][val] += 1
			if rowSet[i][val] > 1 || colSet[k][val] > 1 || boxSet[boxIndex][val] > 1 {
				return false
			}
		}
	}
	return true
}

// 37 解数独
func solveSudoku(board [][]byte) {
}

// 46. 全排列
func permute(nums []int) [][]int {
	var results [][]int
	var track []int
	used := make(map[int]struct{})

	var backTrack func([]int, []int, map[int]struct{}, *[][]int)
	backTrack = func(nums []int, track []int, used map[int]struct{}, results *[][]int) {
		if len(track) == len(nums) {
			result := make([]int, len(track))
			copy(result, track)
			*results = append(*results, result)
			return
		}

		for _, v := range nums {
			if _, ok := used[v]; ok {
				continue
			}

			used[v] = struct{}{}
			track = append(track, v)
			backTrack(nums, track, used, results)
			track = track[:len(track)-1]
			delete(used, v)
		}
	}

	backTrack(nums, track, used, &results)
	return results
}

// 39. 组合总和
func combinationSum(candidates []int, target int) [][]int {
	var results [][]int
	var track []int
	var index int
	sort.Ints(candidates)

	var backTrack func([]int, []int, int, int, *[][]int)
	backTrack = func(cand []int, track []int, sum int, index int, res *[][]int) {
		if sum == target {
			newRes := make([]int, len(track))
			copy(newRes, track)
			*res = append(*res, newRes)
			return
		}

		for k, v := range cand {
			if k < index {
				continue
			}
			sum += v
			if sum > target {
				return
			}
			track = append(track, v)
			index = k
			backTrack(cand, track, sum, index, res)
			track = track[:len(track)-1]
			sum -= v
		}
	}

	backTrack(candidates, track, 0, index, &results)
	return results
}

// 40. 组合总和 II
func combinationSum2(candidates []int, target int) [][]int {
	var results [][]int
	var track []int
	sort.Ints(candidates)

	var backTrack func([]int, []int, int, int, *[][]int)
	backTrack = func(cand []int, track []int, sum int, index int, res *[][]int) {
		for i := index; i < len(cand); i++ {
			if i > index && cand[i] == cand[i-1] {
				index++
				continue
			}
			sum += cand[i]
			if sum > target {
				return
			}
			track = append(track, cand[i])

			if sum == target {
				newRes := make([]int, len(track))
				copy(newRes, track)
				*res = append(*res, newRes)
				return
			}
			backTrack(cand, track, sum, i+1, res)
			track = track[:len(track)-1]
			sum -= cand[i]
		}
	}

	backTrack(candidates, track, 0, 0, &results)
	return results
}

// 41. 缺失的第一个正数
func firstMissingPositive(nums []int) int {
	//只关心[1,len(nums)]之间的正整数
	//先将负数置为len(nums)
	for k, v := range nums {
		if v <= 0 || v > len(nums) {
			nums[k] = len(nums) + 1
		}
	}

	//再用负数做标记，存在正整数n:1<=n<=len(nums),将下标为n-1的值置为负数
	for _, v := range nums {
		index := int(math.Abs(float64(v)) - 1)
		if index >= 0 && index < len(nums) {
			val := nums[index]
			if val > 0 {
				nums[index] = 0 - nums[index]
			}
		}
	}

	for k, v := range nums {
		if v > 0 {
			return k + 1
		}
	}

	return len(nums) + 1
}

// 45. 跳跃游戏 II
func jump(nums []int) int {
	max := func(x, y int) int {
		if x > y {
			return x
		}
		return y
	}

	length := len(nums)
	end := 0
	maxPosition := 0
	steps := 0
	for i := 0; i < length-1; i++ {
		maxPosition = max(maxPosition, i+nums[i])
		if i == end {
			end = maxPosition
			steps++
		}
	}
	return steps
}

// 47. 全排列 II
func permuteUnique(nums []int) [][]int {
	sort.Ints(nums)
	var results [][]int
	var track []int
	used := make([]int, len(nums))

	var backTrack func([]int, []int, []int, int)
	backTrack = func(nums []int, track []int, used []int, index int) {
		if len(track) == len(nums) {
			result := make([]int, len(track))
			copy(result, track)
			results = append(results, result)
			return
		}

		for i := 0; i < len(nums); i++ {
			if used[i] == 1 {
				continue
			}
			if i > 0 && nums[i] == nums[i-1] && used[i-1] == 0 {
				continue
			}
			val := nums[i]
			track = append(track, val)
			used[i] = 1
			backTrack(nums, track, used, i)
			track = track[:len(track)-1]
			used[i] = 0
		}
	}

	backTrack(nums, track, used, 0)
	return results
}

// 48. 旋转图像
func rotate(matrix [][]int) {
	//左右
	width := len(matrix)
	for i := 0; i < width; i++ {
		for j := 0; j < width/2; j++ {
			matrix[i][j], matrix[i][width-j-1] = matrix[i][width-j-1], matrix[i][j]
		}
	}

	for i := 0; i < width-1; i++ {
		for j := 0; j < width-i; j++ {
			matrix[i][j], matrix[width-1-j][width-1-i] = matrix[width-1-j][width-1-i], matrix[i][j]
		}
	}
}

// 53. 最大子数组和
func maxSubArray(nums []int) int {
	length := len(nums)
	if length == 0 {
		return 0
	}
	max := nums[0]
	for i := 1; i < length; i++ {
		if nums[i]+nums[i-1] > nums[i] {
			nums[i] = nums[i] + nums[i-1]
		}
		if nums[i] > max {
			max = nums[i]
		}
	}
	return max
}

// 54. 螺旋矩阵
func spiralOrder(matrix [][]int) []int {
	x, y := 0, len(matrix)-1
	if y >= 0 {
		x = len(matrix[0]) - 1
	}

	u, b, l, r := 0, y, 0, x
	var results []int
	for {
		for i := l; i <= r; i++ {
			results = append(results, matrix[u][i])
		}
		u++
		if u > b {
			break
		}
		for i := u; i <= b; i++ {
			results = append(results, matrix[i][r])
		}
		r--
		if r < l {
			break
		}
		for i := r; i >= l; i-- {
			results = append(results, matrix[b][i])
		}
		b--
		if b < u {
			break
		}
		for i := b; i >= u; i-- {
			results = append(results, matrix[i][l])
		}
		l++
		if l > r {
			break
		}
	}

	return results
}

// 56. 合并区间
func merge(intervals [][]int) [][]int {
	var results [][]int
	bitMap := make([]byte, 20001)

	for _, interval := range intervals {
		l, r := interval[0]*2, interval[1]*2
		for i := l; i <= r; i++ {
			bitMap[i] = 1
		}
	}

	flag := false
	temp := make([]int, 2)
	for k, v := range bitMap {
		if v == 1 {
			if !flag {
				temp[0] = k / 2
				flag = true
			}
		} else {
			if flag {
				temp[1] = (k - 1) / 2
				newInterval := make([]int, 2)
				copy(newInterval, temp)
				results = append(results, newInterval)
				temp = make([]int, 2)
				flag = false
			}
		}
	}

	if flag {
		temp[1] = 10000
		results = append(results, temp)
		flag = false
	}

	return results
}

// 57. 插入区间
func insert(intervals [][]int, newInterval []int) [][]int {
	if len(intervals) == 0 {
		return [][]int{newInterval}
	}

	if len(newInterval) == 0 {
		return intervals
	}

	flag := false
	for index, interval := range intervals {
		if interval[0] < newInterval[0] {
			continue
		}
		intervals = append(intervals[:index], append([][]int{{newInterval[0], newInterval[1]}}, intervals[index:]...)...)
		flag = true
		break
	}
	if !flag {
		intervals = append(intervals, newInterval)
	}

	mergeFn := func(left, right []int) bool {
		if left[1] >= right[0] {
			if left[1] < right[1] {
				left[1] = right[1]
			}
			right[0], right[1] = left[0], left[1]
			return true
		}
		return false
	}

	for i := 0; i < len(intervals)-1; {
		merged := mergeFn(intervals[i], intervals[i+1])
		if merged {
			intervals = append(intervals[:i], intervals[i+1:]...)
		} else {
			i++
		}
	}

	return intervals
}

// 59. 螺旋矩阵 II
func generateMatrix(n int) [][]int {
	results := make([][]int, n)
	for i := 0; i < n; i++ {
		results[i] = make([]int, n)
	}
	x := 1
	l, r, u, b := 0, n-1, 0, n-1
	for {
		for i := l; i <= r; i++ {
			results[u][i] = x
			x++
		}
		u++
		if u > b {
			break
		}
		for i := u; i <= b; i++ {
			results[i][r] = x
			x++
		}
		r--
		if r < l {
			break
		}
		for i := r; i >= l; i-- {
			results[b][i] = x
			x++
		}
		b--
		if b < u {
			break
		}
		for i := b; i >= u; i-- {
			results[i][l] = x
			x++
		}
		l++
		if l > r {
			break
		}
	}

	return results
}

// 78. 子集
func subsets(nums []int) [][]int {
	var results [][]int
	results = append(results, []int{})

	//从右向左遍历nums，每次将results里的值append到nums[i],再加入results
	fn := func(results *[][]int) {
		for i := len(nums) - 1; i >= 0; i-- {
			curLen := len(*results)
			for j := 0; j < curLen; j++ {
				result := append([]int{nums[i]}, (*results)[j]...)
				*results = append(*results, result)
			}
		}
	}

	fn(&results)

	return results
}

// 88. 合并两个有序数组
func merge1(nums1 []int, m int, nums2 []int, n int) {
	i, j, k := 0, 0, 0
	for k < m+n {
		if i == m {
			for n > j {
				nums1[m+n-1-k] = nums2[n-1-j]
				j++
				k++
			}
			break
		}
		if j == n {
			break
		}

		if nums1[m-1-i] > nums2[n-1-j] {
			nums1[m+n-1-k] = nums1[m-1-i]
			i++
		} else {
			nums1[m+n-1-k] = nums2[n-1-j]
			j++
		}
		k++
	}
}

// 90. 子集 II
func subsetsWithDup(nums []int) [][]int {
	sort.Ints(nums)
	var results [][]int
	var path []int
	usedSet := map[int]struct{}{}

	var backTrack func(int, []int, []int, map[int]struct{})
	backTrack = func(index int, nums []int, path []int, usedSet map[int]struct{}) {
		result := make([]int, len(path))
		copy(result, path)
		results = append(results, result)
		if index == len(nums) {
			return
		}

		for i := index; i < len(nums); i++ {
			_, used := usedSet[nums[i]]
			if i > 0 && nums[i] == nums[i-1] && !used {
				continue
			}

			path = append(path, nums[i])
			usedSet[nums[i]] = struct{}{}
			backTrack(i+1, nums, path, usedSet)
			delete(usedSet, nums[i])
			path = path[:len(path)-1]
		}
	}

	backTrack(0, nums, path, usedSet)

	return results
}

// 105. 从前序与中序遍历序列构造二叉树
func buildTree(preorder []int, inorder []int) *TreeNode {
	var makeTreeFn func([]int, []int) *TreeNode
	makeTreeFn = func(preorder []int, inorder []int) *TreeNode {
		if len(preorder) == 0 {
			return nil
		}
		rootVal := preorder[0]
		node := &TreeNode{Val: preorder[0]}
		if len(preorder) > 1 {
			for i := 0; i < len(inorder); i++ {
				if inorder[i] == rootVal {
					node.Left = makeTreeFn(preorder[1:i+1], inorder[:i])
					node.Right = makeTreeFn(preorder[i+1:], inorder[i+1:])
					break
				}
			}
		}
		return node
	}

	return makeTreeFn(preorder, inorder)
}

// 106. 从中序与后序遍历序列构造二叉树
func buildTree1(inorder []int, postorder []int) *TreeNode {
	var makeTreeFn func([]int, []int) *TreeNode
	makeTreeFn = func(inorder []int, postorder []int) *TreeNode {
		if len(postorder) == 0 {
			return nil
		}
		rootVal := postorder[len(postorder)-1]
		node := &TreeNode{Val: rootVal}
		for i := 0; i < len(inorder); i++ {
			if inorder[i] == rootVal {
				node.Right = makeTreeFn(inorder[i+1:], postorder[i:len(postorder)-1])
				node.Left = makeTreeFn(inorder[:i], postorder[:i])

				break
			}
		}
		return node
	}

	return makeTreeFn(inorder, postorder)
}

// 80. 删除有序数组中的重复项 II
func removeDuplicates1(nums []int) int {
	length := len(nums)
	if length <= 2 {
		return length
	}
	length = 2
	for i := 2; i < len(nums); i++ {
		if nums[i] == nums[length-2] {
		} else {
			nums[length] = nums[i]
			length++
		}
	}
	return length
}

// 108. 将有序数组转换为二叉搜索树
func SortedArrayToBST(nums []int) *TreeNode {
	var makeTreeFn func(int, int) *TreeNode
	makeTreeFn = func(l, r int) *TreeNode {
		if r < 0 || l > r {
			return nil
		}
		if l == r {
			return &TreeNode{Val: nums[l]}
		}

		mid := (r + l) / 2
		node := &TreeNode{
			Val:   nums[mid],
			Left:  makeTreeFn(l, mid-1),
			Right: makeTreeFn(mid+1, r),
		}
		return node
	}

	return makeTreeFn(0, len(nums)-1)
}

// 128. 最长连续序列
func longestConsecutive(nums []int) int {
	if len(nums) <= 1 {
		return len(nums)
	}

	//分别记录 头--尾 和 尾--头 迭代更新范围
	start2End := map[int]int{}
	end2Start := map[int]int{}
	for i := 0; i < len(nums); i++ {
		val := nums[i]
		if _, ok := start2End[val]; ok {
			continue
		} else {
			start2End[val] = val
			end2Start[val] = val
		}

		front, behind := val-1, val+1
		if start, ok := end2Start[front]; ok {
			start2End[start] = val
			end2Start[val] = start
			//delete(end2Start, front)
		}

		if end, ok := start2End[behind]; ok {
			start2End[val] = end
			end2Start[end] = val
			//delete(start2End, behind)
		}
		start, ok1 := end2Start[val]
		end, ok2 := start2End[val]
		if ok1 && ok2 {
			start2End[start] = end
			end2Start[end] = start
			//delete(start2End, val)
			//delete(end2Start, val)
		}
	}

	length := 1
	for k, v := range start2End {
		if val := v - k + 1; val > length {
			length = val
		}
	}
	return length
}

// 136. 只出现一次的数字
func singleNumber(nums []int) int {
	num := 0
	for _, v := range nums {
		num = num ^ v
	}
	return num
}

// 509. 斐波那契数
func fib(n int) int {
	if n == 0 || n == 1 {
		return n
	}
	//1.暴力破解
	//return fib(n-1) + fib(n-2)

	//2.备忘录解决[重叠子问题]
	//memo := make([]int, n+1)
	//var dp func([]int, int) int
	//dp = func(memo []int, n int) int {
	//	if n == 0 || n == 1 {
	//		return n
	//	}
	//	if memo[n] > 0 {
	//		return memo[n]
	//	}
	//	memo[n] = dp(memo, n-2) + dp(memo, n-1)
	//	return memo[n]
	//}
	//return dp(memo, n)

	// 3. 数组迭代
	//memo := make([]int, n+1)
	//memo[0], memo[1] = 0, 1
	//for i := 2; i < n+1; i++ {
	//	memo[i] = memo[i-2] + memo[i-1]
	//}
	//return memo[n]

	// 4. 滚动更新优化空间复杂度
	dp1, dp2 := 0, 1
	for i := 2; i < n+1; i++ {
		dpt := dp1 + dp2
		dp1 = dp2
		dp2 = dpt
	}
	return dp2
}

// 322. 零钱兑换
func coinChange(coins []int, amount int) int {
	if amount == 0 {
		return 0
	}
	min := func(a, b int) int {
		if a < b {
			return a
		}
		return b
	}
	dp := make([]int, amount+1)
	dp[0] = 0
	for i := 1; i < amount+1; i++ {
		res := math.MaxInt32
		for _, coin := range coins {
			a := i - coin
			if a >= 0 && dp[a] >= 0 {
				res = min(res, dp[a]+1)
			}
		}
		if res == math.MaxInt32 {
			res = -1
		}
		dp[i] = res
	}
	return dp[amount]
}

// 139. 单词拆分 1 <= s.length <= 300
func wordBreak(s string, wordDict []string) bool {
	length := len(s)
	//对于下标i，值为“子串s[:i+1]的能拼接出来的最长子串点位j”
	dp := make([]bool, length)
	for i := 0; i < length; i++ {
		res := false
		str := s[:i+1]
		for _, word := range wordDict {
			if len(word) > i+1 {
				continue
			}
			if str == word {
				res = true
				break
			}
			if str[i+1-len(word):] == word {
				res = dp[i-len(word)]
				if res == true {
					break
				}
			}
		}
		dp[i] = res
	}
	return dp[length-1]
}

// 152. 乘积最大子数组 1 <= nums.length <= 2 * 104
func maxProduct(nums []int) int {
	maxFn := func(a, b int) int {
		if a > b {
			return a
		}
		return b
	}
	minFn := func(a, b int) int {
		if a > b {
			return b
		}
		return a
	}

	res, dpMax, dpMin := nums[0], nums[0], nums[0]
	for i := 1; i < len(nums); i++ {
		mx, mi := dpMax, dpMin
		dpMax = maxFn(mx*nums[i], maxFn(mi*nums[i], nums[i]))
		dpMin = minFn(mi*nums[i], minFn(mx*nums[i], nums[i]))
		res = maxFn(dpMax, res)
	}

	return res
}

// 153. 寻找旋转排序数组中的最小值
func findMin(nums []int) int {
	var binarySearch func(l, r int) int
	binarySearch = func(l, r int) int {
		if l > r {
			return math.MaxInt32
		}
		if l == r {
			return nums[l]
		}
		mid := (l + r) / 2
		leftMin := binarySearch(l, mid)
		rightMin := binarySearch(mid+1, r)
		if leftMin < rightMin {
			return leftMin
		}
		return rightMin
	}
	return binarySearch(0, len(nums)-1)
}

// 162. 寻找峰值
func findPeakElement(nums []int) int {
	var results []int
	var binarySearch func(l, r int)
	binarySearch = func(l, r int) {
		if l > r {
			return
		}
		mid := (l + r) / 2
		left, right := false, false
		if mid-1 >= 0 {
			if nums[mid] > nums[mid-1] {
				left = true
			}
		} else {
			left = true
		}
		if mid+1 < len(nums) {
			if nums[mid] > nums[mid+1] {
				right = true
			}
		} else {
			right = true
		}
		if left && right {
			results = append(results, mid)
		}
		binarySearch(l, mid-1)
		binarySearch(mid+1, r)
	}

	binarySearch(0, len(nums)-1)
	if len(results) > 0 {
		return results[0]
	}
	return -1
}

// 167. 两数之和 II - 输入有序数组
func twoSum(numbers []int, target int) []int {
	results := make([]int, 2)
	for i, j := 0, len(numbers)-1; i < j; {
		sum := numbers[i] + numbers[j]
		if sum == target {
			results[0], results[1] = i+1, j+1
			break
		} else if sum < target {
			i++
		} else {
			j--
		}
	}
	return results
}

// 169. 多数元素
func majorityElement(nums []int) int {
	num, count := 0, 0
	for _, v := range nums {
		if count == 0 {
			num = v
			count++
		} else {
			if num == v {
				count++
			} else {
				count--
			}
		}
	}
	return num
}

// 179. 最大数
func largestNumber(nums []int) string {
	if len(nums) == 0 {
		return string("0")
	}
	var result string
	strList := make([]string, len(nums))
	for i := 0; i < len(nums); i++ {
		strList[i] = strconv.Itoa(nums[i])
	}
	temp := make([]string, len(nums))

	var lessFunc func(l, r string) bool
	var mergeSort func(nums []string, l, r int, temp []string)
	var mergeFunc func(nums []string, l, mid, r int, temp []string)
	lessFunc = func(l, r string) bool {
		sum1, sum2 := l+r, r+l
		if sum1 > sum2 {
			return false
		}
		return true
	}
	mergeSort = func(nums []string, l, r int, temp []string) {
		if l < r {
			mid := (l + r) / 2
			mergeSort(nums, l, mid, temp)
			mergeSort(nums, mid+1, r, temp)
			mergeFunc(nums, l, mid, r, temp)
		}
	}
	mergeFunc = func(nums []string, l, mid, r int, temp []string) {
		i, j, s := l, mid+1, l
		for i <= mid && j <= r {
			if lessFunc(nums[i], nums[j]) {
				temp[s] = nums[j]
				j++
			} else {
				temp[s] = nums[i]
				i++
			}
			s++
		}
		if i <= mid {
			for ; i <= mid; i++ {
				temp[s] = nums[i]
				s++
			}
		}
		if j <= r {
			for ; j <= r; j++ {
				temp[s] = nums[j]
				s++
			}
		}
		for i := l; i <= r; i++ {
			nums[i] = temp[i]
		}
	}

	mergeSort(strList, 0, len(nums)-1, temp)
	for _, v := range strList {
		if v == "0" && result == "" {
			continue
		}
		result += v
	}
	if result == "" {
		result += "0"
	}
	return result
}

// 189. 轮转数组
func rotate1(nums []int, k int) {
	reverse := func(start, end int) {
		for start < end {
			nums[start], nums[end] = nums[end], nums[start]
			start++
			end--
		}
	}

	reverse(0, len(nums)-1)
	reverse(0, k%len(nums)-1)
	reverse(k%len(nums), len(nums)-1)
}

// 55. 跳跃游戏
func canJump(nums []int) bool {
	dp := make([]bool, len(nums))
	dp[0] = true
	for i := 1; i < len(nums); i++ {
		for j := 0; j < i; j++ {
			if nums[j] >= i-j && dp[j] {
				dp[i] = true
				break
			}
		}
	}
	return dp[len(nums)-1]
}

// 62. 不同路径
func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	dp[0][0] = 1
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i >= 1 {
				dp[i][j] += dp[i-1][j]
			}
			if j >= 1 {
				dp[i][j] += dp[i][j-1]
			}
		}
	}
	return dp[m-1][n-1]
}

// 63. 不同路径 II
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	m, n := len(obstacleGrid), len(obstacleGrid[0])
	if obstacleGrid[0][0] == 0 {
		obstacleGrid[0][0] = -1
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if obstacleGrid[i][j] > 0 {
				continue
			}
			if i >= 1 && obstacleGrid[i-1][j] < 0 {
				obstacleGrid[i][j] += obstacleGrid[i-1][j]
			}
			if j >= 1 && obstacleGrid[i][j-1] < 0 {
				obstacleGrid[i][j] += obstacleGrid[i][j-1]
			}
		}
	}
	if obstacleGrid[m-1][n-1] > 0 {
		return 0
	}

	return -obstacleGrid[m-1][n-1]
}

// 64. 最小路径和
func minPathSum(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			val := grid[i][j]
			if i >= 1 && j >= 1 {
				if grid[i-1][j] > grid[i][j-1] {
					val += grid[i][j-1]
				} else {
					val += grid[i-1][j]
				}
			} else {
				if i >= 1 {
					val += grid[i-1][j]
				} else if j >= 1 {
					val += grid[i][j-1]
				}
			}
			grid[i][j] = val
		}
	}
	return grid[m-1][n-1]
}

// 97. 交错字符串
func isInterleave(s1 string, s2 string, s3 string) bool {
	m, n := len(s1), len(s2)
	if len(s3) != m+n {
		return false
	}
	dp := make([][]byte, m+1)
	for i := 0; i < m+1; i++ {
		dp[i] = make([]byte, n+1)
	}
	dp[0][0] = 1
	for i := 0; i < m+1; i++ {
		for j := 0; j < n+1; j++ {
			idx := i + j - 1
			if i > 0 {
				if dp[i-1][j] == 1 && s1[i-1] == s3[idx] {
					dp[i][j] = 1
				}
			}
			if j > 0 {
				if dp[i][j-1] == 1 && s2[j-1] == s3[idx] {
					dp[i][j] = 1
				}
			}
		}
	}
	if dp[m][n] == 1 {
		return true
	}
	return false
}

// 209. 长度最小的子数组
func minSubArrayLen(target int, nums []int) int {
	res := math.MaxInt32
	start, end := 0, 0
	sum := 0
	for end < len(nums) && end >= start {
		sum += nums[end]
		if sum >= target {
			if res > end-start+1 {
				res = end - start + 1
			}
			sum -= nums[end]
			sum -= nums[start]
			start++
		} else if sum < target {
			end++
		}
	}
	if res == math.MaxInt32 {
		res = 0
	}

	return res
}

// 215. 数组中的第K个最大元素
func findKthLargest(nums []int, k int) int {
	var buildMaxHeap func([]int, int)
	var maxHeapify func([]int, int, int)
	buildMaxHeap = func(nums []int, size int) {
		for i := size / 2; i >= 0; i-- {
			maxHeapify(nums, size, i)
		}
	}
	maxHeapify = func(nums []int, size int, idx int) {
		l, r := idx*2+1, idx*2+2
		maxIdx := idx
		if l < size && nums[l] > nums[maxIdx] {
			maxIdx = l
		}
		if r < size && nums[r] > nums[maxIdx] {
			maxIdx = r
		}
		if maxIdx != idx {
			nums[maxIdx], nums[idx] = nums[idx], nums[maxIdx]
			maxHeapify(nums, size, maxIdx)
		}
	}

	length := len(nums)
	result := 0
	buildMaxHeap(nums, length)
	for i := 0; i < k; i++ {
		result = nums[0]
		nums[0] = nums[length-1]
		length--
		maxHeapify(nums, length, 0)
	}
	return result
}

// 216. 组合总和 III
func combinationSum3(k int, n int) [][]int {
	var results [][]int
	var path []int

	var backTrace func(int, int, []int, *[][]int)
	backTrace = func(idx int, sum int, path []int, results *[][]int) {
		length := len(path)
		if sum == n && length == k {
			result := make([]int, len(path))
			copy(result, path)
			*results = append(*results, result)
		}

		for i := idx; i < 10; i++ {
			sum += i
			path = append(path, i)
			backTrace(i+1, sum, path, results)
			sum -= i
			path = path[:len(path)-1]
		}
	}

	backTrace(1, 0, path, &results)
	return results
}

// 217. 存在重复元素
func containsDuplicate(nums []int) bool {
	sort.Ints(nums)
	for i := 1; i < len(nums); i++ {
		if nums[i] == nums[i-1] {
			return true
		}
	}
	return false
}

// 219. 存在重复元素 II
func containsNearbyDuplicate(nums []int, k int) bool {
	for start, end := 0, 1; start < end && start < len(nums); {
		if k < end-start || end >= len(nums) {
			start++
			end = start + 1
		} else {
			if nums[start] == nums[end] {
				return true
			}
			if end < len(nums) {
				end++
			}
		}
	}

	return false
}

// 221. 最大正方形
func maximalSquare(matrix [][]byte) int {
	m, n := len(matrix), 0
	if m > 0 {
		n = len(matrix[0])
	}
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}

	max := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			dp[i][j] = int(matrix[i][j] - '0')
			if dp[i][j] == 1 {
				if max == 0 {
					max = 1
				}
				if i > 0 && j > 0 {
					if dp[i-1][j] > 0 &&
						dp[i][j-1] > 0 &&
						dp[i-1][j-1] > 0 {
						temp := dp[i-1][j]
						//[i-1][j] == [i][j-1] 的情况
						if dp[i-1][j] == dp[i][j-1] {
							if temp > dp[i-1][j-1] {
								temp = dp[i-1][j-1]
							}
						} else {
							if dp[i-1][j] > dp[i][j-1] {
								temp = dp[i][j-1]
							}
						}
						side := int(math.Sqrt(float64(temp))) + 1
						dp[i][j] = side * side
						if max < dp[i][j] {
							max = dp[i][j]
						}
					}
				}
			}
		}
	}
	return max
}

// 264. 丑数 II
// [1, 2, 3, 4, 5, 6, 8, 9, 10, 12] 是由前 10 个丑数组成的序列。
func nthUglyNumber(n int) int {
	dp := make([]int, n+1)
	dp[1] = 1
	p2, p3, p5 := 1, 1, 1
	for i := 2; i <= n; i++ {
		res2, res3, res5 := dp[p2]*2, dp[p3]*3, dp[p5]*5
		dp[i] = int(math.Min(math.Min(float64(res2), float64(res3)), float64(res5)))
		if dp[i] == dp[p2]*2 {
			p2++
		}
		if dp[i] == dp[p3]*3 {
			p3++
		}
		if dp[i] == dp[p5]*5 {
			p5++
		}
	}
	return dp[n]
}

// 73. 矩阵置零
func setZeroes(matrix [][]int) {
	mSet := map[int]struct{}{}
	nSet := map[int]struct{}{}
	m, n := len(matrix), 0
	if m > 0 {
		n = len(matrix[0])
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if matrix[i][j] == 0 {
				mSet[i] = struct{}{}
				nSet[j] = struct{}{}
			}
		}
	}
	for key := range mSet {
		for i := 0; i < n; i++ {
			matrix[key][i] = 0
		}
	}
	for key := range nSet {
		for i := 0; i < m; i++ {
			matrix[i][key] = 0
		}
	}
}

// 74. 搜索二维矩阵
func searchMatrix(matrix [][]int, target int) bool {
	nums := make([]int, len(matrix))
	for i := 0; i < len(matrix); i++ {
		nums[i] = matrix[i][0]
	}

	var binarySearch func([]int, int, int, int) int
	binarySearch = func(nums []int, start int, end int, target int) int {
		if start >= end {
			if start < len(nums) {
				if nums[start] > target {
					return start - 1
				}
				if nums[start] <= target {
					return start
				}
			} else {
				return -1
			}
		}

		mid := (start + end) / 2
		if nums[mid] == target {
			return mid
		} else if nums[mid] > target {
			return binarySearch(nums, start, mid-1, target)
		} else {
			return binarySearch(nums, mid+1, end, target)
		}
	}
	idx1 := binarySearch(nums, 0, len(nums)-1, target)
	if idx1 < 0 {
		return false
	}
	if matrix[idx1][0] == target {
		return true
	}
	idx2 := binarySearch(matrix[idx1], 1, len(matrix[idx1])-1, target)
	if idx2 > 0 && matrix[idx1][idx2] == target {
		return true
	}
	return false
}

// 912. 排序数组
func sortArray(nums []int) []int {
	var buildMaxHeap func([]int, int)
	var maxHeapify func([]int, int, int)
	var heapSort func([]int)
	buildMaxHeap = func(nums []int, len int) {
		for i := len / 2; i >= 0; i-- {
			maxHeapify(nums, len, i)
		}
	}
	maxHeapify = func(nums []int, len int, idx int) {
		left, right, maxIdx := idx*2+1, idx*2+2, idx
		if left < len && nums[left] > nums[maxIdx] {
			maxIdx = left
		}
		if right < len && nums[right] > nums[maxIdx] {
			maxIdx = right
		}
		if maxIdx == left || maxIdx == right {
			nums[maxIdx], nums[idx] = nums[idx], nums[maxIdx]
			maxHeapify(nums, len, maxIdx)
		}
	}
	heapSort = func(nums []int) {
		length := len(nums)
		buildMaxHeap(nums, length)
		for i := length - 1; i > 0; i-- {
			nums[0], nums[i] = nums[i], nums[0]
			length--
			maxHeapify(nums, length, 0)
		}
	}

	var quickSort func([]int, int, int)
	quickSort = func(nums []int, start, end int) {
		if start >= end {
			return
		}
		target := nums[start]
		i, j := start, end
		for i < j {
			for nums[j] > target && i < j {
				j--
			}
			if i < j {
				nums[i] = nums[j]
				i++
			}
			for nums[i] < target && i < j {
				i++
			}
			if i < j {
				nums[j] = nums[i]
				j--
			}
		}
		nums[i] = target
		quickSort(nums, start, i-1)
		quickSort(nums, i+1, end)
	}

	var mergeFn func([]int, int, int, int, []int)
	var mergeSort func([]int, int, int, []int)
	mergeFn = func(nums []int, start int, mid int, end int, temp []int) {
		i, j, k := start, mid+1, start
		for i <= mid && j <= end {
			if temp[i] > temp[j] {
				nums[k] = temp[j]
				j++
			} else {
				nums[k] = temp[i]
				i++
			}
			k++
		}
		for i <= mid {
			nums[k] = temp[i]
			i++
			k++
		}
		for j <= end {
			nums[k] = temp[j]
			j++
			k++
		}
		for i := start; i <= end; i++ {
			temp[i] = nums[i]
		}
	}
	mergeSort = func(nums []int, start int, end int, temp []int) {
		if start == end {
			temp[start] = nums[start]
			return
		}
		mid := (start + end) / 2
		mergeSort(nums, start, mid, temp)
		mergeSort(nums, mid+1, end, temp)
		mergeFn(nums, start, mid, end, temp)
	}

	heapSort([]int{})
	quickSort([]int{}, 0, 0)
	temp := make([]int, len(nums))
	mergeSort(nums, 0, len(nums)-1, temp)

	return nums
}

// 229. 多数元素 II :尝试设计时间复杂度为 O(n)、空间复杂度为 O(1)的算法
func majorityElement2(nums []int) []int {
	val1, val2 := 0, 0
	count1, count2 := 0, 0
	for i := 0; i < len(nums); i++ {
		if nums[i] == val1 && count1 > 0 {
			count1++
		} else if nums[i] == val2 && count2 > 0 {
			count2++
		} else if count1 == 0 {
			val1 = nums[i]
			count1++
		} else if count2 == 0 {
			val2 = nums[i]
			count2++
		} else {
			count1--
			count2--
		}
	}

	c1, c2 := 0, 0
	for i := 0; i < len(nums); i++ {
		if nums[i] == val1 && count1 > 0 {
			c1++
		}
		if val2 != val1 && nums[i] == val2 && count2 > 0 {
			c2++
		}
	}

	var results []int
	if c1 > len(nums)/3 {
		results = append(results, val1)
	}
	if c2 > len(nums)/3 {
		results = append(results, val2)
	}

	return results
}

// 198. 打家劫舍
func rob(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	max_1, dp_1, dp_2 := 0, 0, 0
	result := 0
	for i := 0; i < len(nums); i++ {
		dp_2 = max_1 + nums[i]
		if dp_1 > max_1 {
			max_1 = dp_1
		}
		if dp_2 > result {
			result = dp_2
		}
		dp_1 = dp_2
	}
	return result
}

// 213. 打家劫舍 II
func rob2(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	max_1, dp_1, dp_2 := 0, 0, 0
	result := 0
	for i := 0; i < len(nums)-1; i++ {
		dp_2 = max_1 + nums[i]
		if dp_1 > max_1 {
			max_1 = dp_1
		}
		if dp_2 > result {
			result = dp_2
		}
		dp_1 = dp_2
	}

	max_1, dp_1, dp_2 = 0, 0, 0
	for i := 1; i < len(nums); i++ {
		dp_2 = max_1 + nums[i]
		if dp_1 > max_1 {
			max_1 = dp_1
		}
		if dp_2 > result {
			result = dp_2
		}
		dp_1 = dp_2
	}

	return result
}

// 238. 除自身以外数组的乘积
func productExceptSelf(nums []int) []int {
	if len(nums) == 0 {
		return nil
	}
	l, r := make([]int, len(nums)), make([]int, len(nums))
	l[0], r[len(nums)-1] = 1, 1
	for i := 1; i < len(nums); i++ {
		l[i] = l[i-1] * nums[i-1]
	}
	for i := len(nums) - 2; i >= 0; i-- {
		r[i] = r[i+1] * nums[i+1]
	}
	for i := 0; i < len(nums); i++ {
		l[i] = l[i] * r[i]
	}
	return l
}

// 240. 搜索二维矩阵 II
func searchMatrix2(matrix [][]int, target int) bool {
	var binarySearch func([]int, int, int, int) (int, bool)
	binarySearch = func(nums []int, start int, end int, target int) (int, bool) {
		if start >= end {
			if start < len(nums) {
				if nums[start] == target {
					return start, true
				} else {
					return start, false
				}
			} else {
				return start, false
			}
		}
		mid := (start + end) / 2
		if nums[mid] == target {
			return mid, true
		} else if nums[mid] > target {
			return binarySearch(nums, start, mid-1, target)
		} else {
			return binarySearch(nums, mid+1, end, target)
		}
	}

	m, n := len(matrix), 0
	if m > 0 {
		n = len(matrix[0])
	}

	x, y := 0, 0
	for {
		w := m
		if m > n {
			w = n
		}
		//对角线
		nums := make([]int, w)
		for i := 0; i < w; i++ {
			nums[i] = matrix[i+x][i+y]
		}
		idx, ok := binarySearch(nums, 0, len(nums)-1, target)
		if ok {
			return ok
		}
		if idx >= 0 && idx <= len(nums) {
			for i := 0; i < m; i++ {
				for j := idx; j < n; j++ {
					if matrix[i][j] == target {
						return true
					}
				}
			}
			for i := idx; i < m; i++ {
				for j := 0; j < n; j++ {
					if matrix[i][j] == target {
						return true
					}
				}
			}
		}

		//划分区域
		if m == n {
			break
		} else if m > n {
			x += w
			m = m - n
		} else {
			y += w
			n = n - m
		}
	}

	return false
}

// 228. 汇总区间
func summaryRanges(nums []int) []string {
	var intervals []string
	if len(nums) == 0 {
		return intervals
	}

	start, end := nums[0], nums[0]

	rev := false
	for i := 1; i < len(nums); i++ {
		if nums[i] < nums[i-1] {
			rev = true
		}
		if !rev && nums[i]-nums[i-1] <= 1 {
			end++
			continue
		}
		if rev && nums[i-1]-nums[i] <= 1 {
			start--
			continue
		}

		interval := fmt.Sprintf("%d->%d", start, end)
		if start == end {
			interval = fmt.Sprintf("%d", start)
		}
		intervals = append(intervals, interval)
		start, end = nums[i], nums[i]
	}

	interval := fmt.Sprintf("%d->%d", start, end)
	if start == end {
		interval = fmt.Sprintf("%d", start)
	}
	intervals = append(intervals, interval)

	return intervals
}

// 268. 丢失的数字
func missingNumber(nums []int) int {
	sum := 0
	for i := 0; i < len(nums); i++ {
		sum += i + 1
		sum -= nums[i]
	}
	return sum
}

// 283. 移动零
func moveZeroes(nums []int) {
	for i := 0; i < len(nums); i++ {
		for j := 0; j < len(nums)-1; j++ {
			if nums[j] == 0 {
				nums[j], nums[j+1] = nums[j+1], nums[j]
			}
		}
	}
}

// 287. 寻找重复数
// 不修改数组 nums 且空间复杂度O(1)； 时间复杂度O(n)；
func findDuplicate(nums []int) (result int) {
	p, q := 0, 0
	for len(nums) > 0 {
		p = nums[p]
		q = nums[nums[q]]
		if p == q {
			break
		}
	}
	q = 0
	for len(nums) > 0 && p != q {
		p = nums[p]
		q = nums[q]
	}
	result = p
	return
}

// 2582. 递枕头
func passThePillow(n int, time int) int {
	if n == 1 {
		return 1
	}
	direct := true
	i, j := 1, 1
	for ; i <= time; i++ {
		if direct && j == n || !direct && j == 1 {
			direct = !direct
		}
		if direct {
			j++
		} else {
			j--
		}
	}
	return j
}
