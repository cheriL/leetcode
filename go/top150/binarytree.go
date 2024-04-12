package top150

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
	return nil
}

func Flatten(root *TreeNode) {
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
