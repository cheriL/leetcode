package top150

import "math"

func maxDepth(root *TreeNode) int {
	var traversal func(*TreeNode, int) int
	traversal = func(node *TreeNode, count int) int {
		if node == nil {
			return count
		}

		count++
		l, r := traversal(node.Left, count), traversal(node.Right, count)
		if l > r {
			return l
		}
		return r
	}

	return traversal(root, 0)
}

func isSameTree(node *TreeNode, node2 *TreeNode) bool {
	switch {
	case node == nil && node2 == nil:
		return true
	case node != nil && node2 != nil:
		if node.Val != node2.Val {
			return false
		}
	default:
		return false
	}
	return isSameTree(node.Left, node2.Left) && isSameTree(node.Right, node2.Right)
}

func invertTree(root *TreeNode) *TreeNode {
	if root != nil {
		root.Left, root.Right = root.Right, root.Left
		invertTree(root.Left)
		invertTree(root.Right)
	}

	return root
}

func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}
	var traversal func(node *TreeNode, node2 *TreeNode) bool
	traversal = func(node *TreeNode, node2 *TreeNode) bool {
		if node == nil && node2 == nil {
			return true
		} else if node != nil && node2 != nil {
			if node.Val != node2.Val {
				return false
			}
			return traversal(node.Left, node2.Right) && traversal(node.Right, node2.Left)
		}
		return false
	}

	return traversal(root.Left, root.Right)
}

func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}

	node := &TreeNode{Val: preorder[0]}
	for k, v := range inorder {
		if v == node.Val {
			node.Left = buildTree(preorder[1:k+1], inorder[:k])
			node.Right = buildTree(preorder[k+1:], inorder[k+1:])
			break
		}
	}
	return node
}

func buildTree2(inorder []int, postorder []int) *TreeNode {
	if len(postorder) == 0 {
		return nil
	}
	node := &TreeNode{Val: postorder[len(postorder)-1]}
	for i := 0; i < len(inorder); i++ {
		if inorder[i] == node.Val {
			node.Left = buildTree2(inorder[:i], postorder[:i])
			node.Right = buildTree2(inorder[i+1:], postorder[i:len(postorder)-1])
		}
	}
	return node
}

func connect(root *Node) *Node {
	if root == nil {
		return root
	}

	var rNode *Node
	for next := root.Next; next != nil; next = next.Next {
		if next.Left != nil {
			rNode = next.Left
			break
		}
		if next.Right != nil {
			rNode = next.Right
			break
		}
	}
	if root.Right != nil {
		root.Right.Next = rNode
		rNode = root.Right
	}
	if root.Left != nil {
		root.Left.Next = rNode
	}
	root.Right = connect(root.Right)
	root.Left = connect(root.Left)

	return root
}

func flatten(root *TreeNode) {
	var traversal func(root *TreeNode) *TreeNode
	traversal = func(root *TreeNode) *TreeNode {
		if root == nil || root.Left == nil && root.Right == nil {
			return root
		}
		left, right := root.Left, root.Right
		leftLeaf, rightLeaf := traversal(root.Left), traversal(root.Right)
		if leftLeaf != nil {
			leftLeaf.Right = right
			root.Right = left
			root.Left = nil
		}
		if rightLeaf == nil {
			rightLeaf = leftLeaf
		}
		return rightLeaf
	}

	traversal(root)
}

func sumNumbers(root *TreeNode) int {
	var result int
	var traversal func(node *TreeNode, num int)
	traversal = func(node *TreeNode, num int) {
		if node == nil {
			return
		}
		num = num*10 + node.Val
		if node.Left == nil && node.Right == nil {
			result += num
			return
		}
		traversal(node.Left, num)
		traversal(node.Right, num)
	}

	traversal(root, 0)
	return result
}

// [-1,5,null,4,null,null,2,-4]
// 2, -1
func maxPathSum(root *TreeNode) int {
	result := math.MinInt32

	var traversalTree func(node *TreeNode) int
	traversalTree = func(node *TreeNode) int {
		if node == nil {
			return 0
		}

		var sum int
		left := traversalTree(node.Left)
		right := traversalTree(node.Right)
		switch {
		case node.Left == nil && node.Right == nil:
			sum = node.Val
		case node.Left != nil && node.Right != nil:
			if temp := left + right + node.Val; temp > result {
				result = temp
			}
			if left > right {
				sum = left + node.Val
			} else {
				sum = right + node.Val
			}
		case node.Left == nil && node.Right != nil:
			sum = right + node.Val
		case node.Left != nil && node.Right == nil:
			sum = left + node.Val
		}

		if sum > node.Val {
			if sum > result {
				result = sum
			}
			return sum
		}
		if node.Val > result {
			result = node.Val
		}
		return node.Val
	}

	traversalTree(root)
	return result
}