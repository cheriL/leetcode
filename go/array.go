//https://leetcode.cn/tag/array/problemset/

package _go

import (
	"math"
	"sort"
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
