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
