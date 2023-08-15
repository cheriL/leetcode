package _go

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
