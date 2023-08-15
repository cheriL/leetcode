//https://leetcode.cn/tag/array/problemset/

package _go

import "sort"

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

// 37
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
