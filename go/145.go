package _go

func postorderTraversal(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}

	if root.Left == nil && root.Right == nil {
		res = append(res, root.Val)
	} else {
		leftRes := postorderTraversal(root.Left)
		rightRes := postorderTraversal(root.Right)
		res = append(res, leftRes...)
		res = append(res, rightRes...)
		res = append(res, root.Val)
	}
	return res
}
