package top150

import (
	"math"
	"sort"
)

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

//func countNodes(root *TreeNode) int {
//	if root == nil {
//		return 0
//	}
//
//	// 完全二叉树，最左节点一定在最后一层
//	h := 0
//	for node := root.Left; node != nil; node = node.Left {
//		h++
//	}
//	// 节点的个数，一定在 [2^h, 2^(h+1)-1]，在这个范围内，判断对应的节点是否存在
//	// 最后一层的第 k 个节点，k 值的二进制表示这个节点在树上的路径， 0为左子节点，1为右子节点
//	findNode := func(k int) bool {
//		for level := h; level > 0; level-- {
//
//		}
//
//		return true
//	}
//
//	var binarySearch func(int, int) int
//	binarySearch = func(start, end int) int {
//		for start < end {
//			mid := (start + end) >> 1
//			if findNode(mid) {
//				start = mid + 1
//			} else {
//				end = mid
//			}
//		}
//		return start
//	}
//
//	sort.Search()
//}

func countNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}

	// 完全二叉树，最左节点一定在最后一层
	level := 0
	for node := root; node.Left != nil; node = node.Left {
		level++
	}

	// 节点的个数，一定在 [2^h, 2^(h+1)-1]，在这个范围内，判断对应的节点是否存在
	// 最后一层的第 k 个节点，k 的二进制表示包含 h+1 位，
	// 其中最高位是 1，其余各位从高到低表示从根节点到第 k 个节点的路径，
	//0 表示移动到左子节点，1 表示移动到右子节点。通过位运算得到第 k 个节点对应的路径，
	//判断该路径对应的节点是否存在，即可判断第 k 个节点是否存在
	return sort.Search(1<<(level+1), func(k int) bool {
		if k <= 1<<level {
			return false
		}
		bits := 1 << (level - 1)
		node := root
		for node != nil && bits > 0 {
			if bits&k == 0 {
				node = node.Left
			} else {
				node = node.Right
			}
			bits >>= 1
		}
		return node == nil
	}) - 1
}

func lowestCommonAncestor(root, p, q *TreeNode) (result *TreeNode) {
	var findCommonAncestor func(node *TreeNode) int
	findCommonAncestor = func(node *TreeNode) int {
		if node == nil {
			return 0
		}

		left := findCommonAncestor(node.Left)
		right := findCommonAncestor(node.Right)
		if node.Val == p.Val || node.Val == q.Val {
			if left > 0 || right > 0 {
				result = node
				return 2
			} else {
				return 1
			}
		}

		if left == 1 && right == 1 {
			result = node
			return 2
		} else {
			return left + right
		}
	}

	findCommonAncestor(root)
	return
}

func rightSideView(root *TreeNode) (results []int) {
	if root == nil {
		return
	}

	nodes := []*TreeNode{root}
	for len(nodes) > 0 {
		length := len(nodes)
		results = append(results, nodes[length-1].Val)
		for i := 0; i < length; i++ {
			if nodes[i].Left != nil {
				nodes = append(nodes, nodes[i].Left)
			}
			if nodes[i].Right != nil {
				nodes = append(nodes, nodes[i].Right)
			}
		}
		nodes = nodes[length:]
	}
	return
}

func averageOfLevels(root *TreeNode) (results []float64) {
	if root == nil {
		return
	}

	nodes := []*TreeNode{root}
	for len(nodes) > 0 {
		length := len(nodes)
		sum := float64(0)
		for i := 0; i < length; i++ {
			if nodes[i].Left != nil {
				nodes = append(nodes, nodes[i].Left)
			}
			if nodes[i].Right != nil {
				nodes = append(nodes, nodes[i].Right)
			}
			sum += float64(nodes[i].Val)
		}
		results = append(results, sum/float64(length))
		nodes = nodes[length:]
	}
	return
}

func levelOrder(root *TreeNode) (results [][]int) {
	if root == nil {
		return
	}

	nodes := []*TreeNode{root}
	for len(nodes) > 0 {
		length := len(nodes)
		result := make([]int, length)
		for i := 0; i < length; i++ {
			if nodes[i].Left != nil {
				nodes = append(nodes, nodes[i].Left)
			}
			if nodes[i].Right != nil {
				nodes = append(nodes, nodes[i].Right)
			}
			result[i] = nodes[i].Val
		}
		results = append(results, result)
		nodes = nodes[length:]
	}
	return
}

func zigzagLevelOrder(root *TreeNode) (results [][]int) {
	if root == nil {
		return
	}

	nodes := []*TreeNode{root}
	adverse := true
	for ; len(nodes) > 0; adverse = !adverse {
		length := len(nodes)
		result := make([]int, length)
		for i := 0; i < length; i++ {
			if nodes[i].Left != nil {
				nodes = append(nodes, nodes[i].Left)
			}
			if nodes[i].Right != nil {
				nodes = append(nodes, nodes[i].Right)
			}

			if adverse {
				result[i] = nodes[i].Val
			} else {
				result[i] = nodes[length-1-i].Val
			}
		}
		results = append(results, result)
		nodes = nodes[length:]
	}
	return
}

func getMinimumDifference(root *TreeNode) int {
	var nodeVal []int
	var traversal func(node *TreeNode)
	traversal = func(node *TreeNode) {
		if node == nil {
			return
		}
		traversal(node.Left)
		nodeVal = append(nodeVal, node.Val)
		traversal(node.Right)
	}

	traversal(root)
	result := math.MaxInt32
	for i := 1; i < len(nodeVal); i++ {
		if val := nodeVal[i] - nodeVal[i-1]; val < result {
			result = val
		}
	}

	return result
}
