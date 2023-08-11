package _go

/*
*94 中序遍历
  - Definition for a binary tree node.
*/
func inorderTraversal(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}

	if root.Left == nil && root.Right == nil {
		res = append(res, root.Val)
	} else {
		leftRes := inorderTraversal(root.Left)
		rightRes := inorderTraversal(root.Right)
		res = append(res, leftRes...)
		res = append(res, root.Val)
		res = append(res, rightRes...)
	}
	return res
}

// 95生成 不同的二叉搜索树 II  回溯
func generateTrees(n int) []*TreeNode {
	var res []*TreeNode

	var gen func(int, int) []*TreeNode
	gen = func(start int, end int) []*TreeNode {
		if start > end {
			return []*TreeNode{nil}
		}
		var list []*TreeNode

		for i := start; i <= end; i++ {
			leftRes := gen(start, i-1)
			rightRes := gen(i+1, end)
			for _, lr := range leftRes {
				for _, rr := range rightRes {
					newNode := &TreeNode{
						Val:   i,
						Left:  lr,
						Right: rr,
					}
					list = append(list, newNode)
				}
			}
		}
		return list
	}

	res = gen(1, n)
	return res
}

// 96生成 不同的二叉搜索树数量 动态规划
func numTrees(n int) int {
	G := make([]int, n+1)
	G[0], G[1] = 1, 1

	//n为长度，G[n]为生成二叉树的数量
	for i := 2; i <= n; i++ {
		sum := 0
		//j根节点
		for j := 1; j <= i; j++ {
			//左子树的数量为G(i-1)
			//右子树的数量为G(n-i)
			left := G[j-1]
			right := G[i-j]
			sum += left * right
		}
		G[i] = sum
	}
	return G[n]
}
