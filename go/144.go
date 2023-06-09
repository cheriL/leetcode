package _go


func preorderTraversal(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}

	if root.Left == nil && root.Right == nil {
		res = append(res, root.Val)
	} else {
		leftRes := preorderTraversal(root.Left)
		rightRes := preorderTraversal(root.Right)
		res = append(res, root.Val)
		res = append(res, leftRes...)
		res = append(res, rightRes...)
	}
	return res
}