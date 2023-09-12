//https://leetcode.cn/tag/binary-tree/problemset/

package _go

import "math"

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

func isValidBST(root *TreeNode) bool {
	var helper func(node *TreeNode, min, max int) bool
	helper = func(node *TreeNode, min, max int) bool {
		if node == nil {
			return true
		}
		if node.Val <= min || node.Val >= max {
			return false
		}
		return helper(node.Left, min, node.Val) && helper(node.Right, node.Val, max)
	}

	return helper(root, math.MinInt64, math.MaxInt64)
}

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

// 100. 相同的树
func isSameTree(p *TreeNode, q *TreeNode) bool {
	var validate func(*TreeNode, *TreeNode) bool
	validate = func(node *TreeNode, node2 *TreeNode) bool {
		if node == nil && node2 == nil {
			return true
		} else if node != nil && node2 != nil {
			return node.Val == node2.Val &&
				validate(node.Left, node2.Left) &&
				validate(node.Right, node2.Right)
		} else {
			return false
		}
	}

	return validate(p, q)
}

// 101. 对称二叉树
func isSymmetric(root *TreeNode) bool {
	var validate func(*TreeNode, *TreeNode) bool
	validate = func(node *TreeNode, node2 *TreeNode) bool {
		if node == nil && node2 == nil {
			return true
		} else if node != nil && node2 != nil {
			return node.Val == node2.Val &&
				validate(node.Left, node2.Right) &&
				validate(node.Right, node2.Left)
		} else {
			return false
		}
	}

	return validate(root.Left, root.Right)
}

// 102. 二叉树的层序遍历
func levelOrder(root *TreeNode) [][]int {
	var results [][]int
	if root == nil {
		return results
	}

	queue := []*TreeNode{root}
	for len(queue) > 0 {
		length := len(queue)
		result := make([]int, length)
		for i := 0; i < length; i++ {
			result[i] = queue[i].Val
			if queue[i].Left != nil {
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				queue = append(queue, queue[i].Right)
			}
		}
		results = append(results, result)
		queue = queue[length:]
	}
	return results
}

// 103. 二叉树的锯齿形层序遍历
func zigzagLevelOrder(root *TreeNode) [][]int {
	var results [][]int
	if root == nil {
		return results
	}

	queue := []*TreeNode{root}
	for rev := false; len(queue) > 0; rev = !rev {
		length := len(queue)
		result := make([]int, length)
		for i := 0; i < length; i++ {
			if rev {
				result[i] = queue[length-1-i].Val
			} else {
				result[i] = queue[i].Val
			}
			if queue[i].Left != nil {
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				queue = append(queue, queue[i].Right)
			}
		}
		results = append(results, result)
		queue = queue[length:]
	}
	return results
}
