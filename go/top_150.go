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

// 26. 删除有序数组中的重复项 1 1 1 2
func removeDuplicates2(nums []int) int {
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