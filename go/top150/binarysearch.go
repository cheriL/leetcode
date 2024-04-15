package top150

func searchInsert(nums []int, target int) int {
	var binarySearch func([]int, int, int, int) int
	binarySearch = func(nums []int, start int, end int, target int) int {
		if start == end {
			if nums[start] < target {
				return start + 1
			}
			return start
		}
		mid := (start + end) / 2
		if nums[mid] == target {
			return mid
		} else if nums[mid] > target {
			return binarySearch(nums, start, mid, target)
		}
		return binarySearch(nums, mid+1, end, target)
	}

	return binarySearch(nums, 0, len(nums)-1, target)
}

func hasPathSum(root *TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}

	sum := targetSum - root.Val
	if root.Left == nil && root.Right == nil {
		if sum == 0 {
			return true
		}
		return false
	}

	left, right := false, false
	if root.Left != nil {
		left = hasPathSum(root.Left, sum)
	}
	if root.Right != nil {
		right = hasPathSum(root.Right, sum)
	}

	return left || right
}
