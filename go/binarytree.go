//https://leetcode.cn/tag/binary-tree/problemset/

package _go

import (
	"math"
)

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

// 104. 二叉树的最大深度
func maxDepth(root *TreeNode) int {
	var traversal func(*TreeNode) int
	traversal = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		lDepth := traversal(node.Left)
		rDepth := traversal(node.Right)
		if lDepth > rDepth {
			return lDepth + 1
		}
		return rDepth + 1
	}
	return traversal(root)
}

// 107. 二叉树的层序遍历 II
func levelOrderBottom(root *TreeNode) [][]int {
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
		results = append([][]int{result}, results...)
		queue = queue[length:]
	}
	return results
}

// 1110. 删点成林
func delNodes(root *TreeNode, to_delete []int) []*TreeNode {
	var results []*TreeNode
	targetSet := map[int]struct{}{}
	for _, v := range to_delete {
		targetSet[v] = struct{}{}
	}

	var traversal func(*TreeNode, *TreeNode)
	traversal = func(node *TreeNode, parent *TreeNode) {
		if node == nil {
			return
		}
		if _, ok := targetSet[node.Val]; ok {
			if parent != nil {
				if parent.Left == node {
					parent.Left = nil
				} else {
					parent.Right = nil
				}
			}
			traversal(node.Left, nil)
			traversal(node.Right, nil)
			node = nil
		} else {
			if parent == nil {
				results = append(results, node)
			}
			traversal(node.Left, node)
			traversal(node.Right, node)
		}
	}
	traversal(root, nil)
	return results
}

// 110. 平衡二叉树
func isBalanced(root *TreeNode) bool {
	var traversal func(*TreeNode) (bool, int)
	traversal = func(node *TreeNode) (bool, int) {
		if node == nil {
			return true, 0
		}
		ok, l := traversal(node.Left)
		ok2, r := traversal(node.Right)
		if l > r {
			return ok && ok2 && l == r+1, l + 1
		} else if l == r {
			return ok && ok2, l + 1
		} else {
			return ok && ok2 && r == l+1, r + 1
		}
	}

	ok, _ := traversal(root)
	return ok
}

// 112. 路径总和
func hasPathSum(root *TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}
	var traversal func(*TreeNode, int) bool
	traversal = func(node *TreeNode, val int) bool {
		if node == nil {
			return false
		}
		sum := node.Val + val
		if node.Left == nil && node.Right == nil {
			if sum == targetSum {
				return true
			}
		}

		return traversal(node.Left, sum) || traversal(node.Right, sum)
	}
	return traversal(root, 0)
}

// 113. 路径总和 II
func pathSum(root *TreeNode, targetSum int) [][]int {
	if root == nil {
		return nil
	}

	var results [][]int
	var result []int

	var traversal func(*TreeNode, int, []int)
	traversal = func(node *TreeNode, val int, result []int) {
		if node == nil {
			return
		}

		path := make([]int, len(result)+1)
		copy(path, result)
		path[len(result)] = node.Val

		sum := node.Val + val
		if node.Left == nil && node.Right == nil {
			if sum == targetSum {
				results = append(results, path)
				return
			}
		}

		traversal(node.Left, sum, path)
		traversal(node.Right, sum, path)
	}
	traversal(root, 0, result)

	return results
}

// 124. 二叉树中的最大路径和
func maxPathSum(root *TreeNode) int {
	if root == nil {
		return 0
	}
	max := root.Val
	var findMax func(*TreeNode) int
	findMax = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		//该节点下的最大，可能是 node + left + right
		sum := node.Val
		leftVal := findMax(node.Left)
		rightVal := findMax(node.Right)
		if sum+leftVal > sum {
			sum += leftVal
		}
		if sum+rightVal > sum {
			sum += rightVal
		}
		if sum > max {
			max = sum
		}
		// 只返回 node + left 或 node + right 或 node
		maxOfChild := leftVal
		if leftVal < rightVal {
			maxOfChild = rightVal
		}
		if node.Val+maxOfChild > node.Val {
			return node.Val + maxOfChild
		}
		return node.Val
	}
	findMax(root)
	return max
}

// 129. 求根节点到叶节点数字之和
func sumNumbers(root *TreeNode) int {
	result := 0
	var sumFn func(*TreeNode, int)
	sumFn = func(node *TreeNode, pVal int) {
		if node == nil {
			return
		}
		val := pVal*10 + node.Val
		if node.Left == nil && node.Right == nil {
			result += val
			return
		}
		if node.Left != nil {
			sumFn(node.Left, val)
		}
		if node.Right != nil {
			sumFn(node.Right, val)
		}
	}
	sumFn(root, 0)
	return result
}
