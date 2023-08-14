// Package _go https://leetcode.cn/tag/linked-list/problemset/
package _go

// 86. 分隔链表
func partition(head *ListNode, x int) *ListNode {
	dummyNode1 := &ListNode{
		Val:  -1,
		Next: nil,
	}
	dummyNode2 := &ListNode{
		Val:  -1,
		Next: nil,
	}

	p1, p2 := dummyNode1, dummyNode2
	for head != nil {
		if head.Val < x {
			p1.Next = &ListNode{
				Val:  head.Val,
				Next: nil,
			}
			p1 = p1.Next
		} else {
			p2.Next = &ListNode{
				Val:  head.Val,
				Next: nil,
			}
			p2 = p2.Next
		}
		head = head.Next
	}

	p1.Next = dummyNode2.Next

	return dummyNode1.Next
}

//61旋转链表
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	length, end := 1, head
	for end.Next != nil {
		length += 1
		end = end.Next
	}

	k1 := k % length
	if k1 == 0 {
		return head
	}
	i, temp := 1, head
	var newHead *ListNode
	for {
		if i >= length-k1 {
			newHead = temp.Next
			temp.Next = nil
			break
		}

		i++
		temp = temp.Next
	}

	if end != nil {
		end.Next = head
	}

	return newHead
}

// 82. 删除排序链表中的重复元素 II
func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}

	dummyNode := &ListNode{}
	pre, p, q := dummyNode, head, head.Next
	for q != nil {
		if q.Val > p.Val {
			if p.Next == q {
				pre.Next = p
				pre = pre.Next
				pre.Next = nil
			}
			p = q
			q = q.Next
		} else {
			q = q.Next
		}
	}

	//保留尾节点情况
	if p.Next == nil {
		pre.Next = p
	}
	return dummyNode.Next
}

// 92. 反转链表 II
func reverseBetween(head *ListNode, left int, right int) *ListNode {
	if head == nil || left == right {
		return head
	}

	pre := head
	var head2 *ListNode
	if left > 1 {
		for i := 1; i < left-1 && pre != nil; i++ {
			pre = pre.Next
		}
		head2 = pre.Next
	} else {
		head2 = pre
	}

	if head2 == nil || head2.Next == nil {
		return head
	}

	var p *ListNode
	var q *ListNode
	p, q = head2, head2.Next
	p.Next = nil
	for i := 1; i <= right-left && p != nil; i++ {
		temp := q.Next
		q.Next = p
		p = q
		q = temp
	}
	head2.Next = q
	if head2 == head {
		return p
	} else {
		pre.Next = p
		return head
	}
}

// 109. 有序链表转换二叉搜索树
func sortedListToBST(head *ListNode) *TreeNode {
	var findRoot func(*ListNode, *ListNode) *TreeNode

	findRoot = func(start, end *ListNode) *TreeNode {
		if start == nil {
			return nil
		}

		if start == end {
			return &TreeNode{
				Val: start.Val,
			}
		}

		if start.Next == end {
			if end == nil {
				return &TreeNode{
					Val: start.Val,
				}
			} else {
				return &TreeNode{
					Val: end.Val,
					Left: &TreeNode{
						Val: start.Val,
					},
				}
			}
		}

		p, q := start, start
		//找到root节点时的前一个节点
		var pre *ListNode
		for q != end && q != nil && q.Next != nil {
			pre = p
			p = p.Next
			q = q.Next.Next
		}
		//断链
		if pre != nil {
			pre.Next = nil
		}

		root := &TreeNode{
			Val: p.Val,
		}
		root.Left = findRoot(start, pre)
		root.Right = findRoot(p.Next, q)

		return root
	}

	return findRoot(head, nil)
}

// 114. 二叉树展开为链表
func flatten(root *TreeNode) {
	var stack []*TreeNode
	isEmpty := func() bool {
		return len(stack) == 0
	}
	pop := func() *TreeNode {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		return node
	}
	push := func(node *TreeNode) {
		stack = append(stack, node)
	}

	var chgFunc func(node *TreeNode)
	chgFunc = func(node *TreeNode) {
		if node == nil {
			return
		}
		//右子树入栈
		if node.Right != nil {
			temp := node.Right
			push(temp)
			node.Right = nil
		}
		if node.Left != nil {
			node.Right = node.Left
			node.Left = nil
		} else {
			if !isEmpty() {
				child := pop()
				node.Right = child
			}
		}
		if node.Right == nil {
			return
		}
		chgFunc(node.Right)
	}

	chgFunc(root)
}

// 83. 删除排序链表中的重复元素
func deleteDuplicates1(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}

	p := head
	for p.Next != nil {
		if p.Next.Val == p.Val {
			temp := p.Next
			p.Next = temp.Next
			temp.Next = nil
		} else {
			p = p.Next
		}
	}

	return head
}

// 116. 填充每个节点的下一个右侧节点指针
func connect(root *Node) *Node {
	//先序地去遍历树，过程中利用 next
	var doConn func(*Node)
	doConn = func(node *Node) {
		if node == nil {
			return
		}
		if node.Left != nil {
			node.Left.Next = node.Right
			//完美二叉树，有左子节点的情况一定也有右子节点
			if node.Next != nil {
				node.Right.Next = node.Next.Left
			}
		}
		doConn(node.Left)
		doConn(node.Right)
	}

	doConn(root)
	return root
}

// 117. 填充每个节点的下一个右侧节点指针 II
func connect2(root *Node) *Node {
	var doConn func(*Node)
	doConn = func(node *Node) {
		if node == nil {
			return
		}

		// [1,2,3,4,5,null,6,7,null,null,null,null,8]
		// 拿到本层右边子树的第一个不为nil的节点
		var rightNode *Node
		next := node.Next
		for next != nil {
			if next.Left != nil {
				rightNode = next.Left
				break
			}
			if next.Right != nil {
				rightNode = next.Right
				break
			}
			next = next.Next
		}

		if node.Right != nil {
			node.Right.Next = rightNode
			rightNode = node.Right
		}
		if node.Left != nil {
			node.Left.Next = rightNode
		}

		doConn(node.Right)
		doConn(node.Left)
	}

	doConn(root)
	return root
}

// 141. 环形链表
func hasCycle(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}
	if head.Next == head {
		return true
	}

	dummyNode := &ListNode{Next: head}
	for p, q := dummyNode.Next, dummyNode.Next.Next; q != nil && q.Next != nil; {
		if q == p || q.Next == p {
			return true
		}
		p = p.Next
		q = q.Next.Next
	}

	return false
}

// 142. 环形链表 II
func detectCycle(head *ListNode) *ListNode {
	nodeSet := make(map[*ListNode]struct{})
	p := head
	for p != nil {
		if _, ok := nodeSet[p]; ok {
			return p
		}
		nodeSet[p] = struct{}{}
		p = p.Next
	}
	return nil
}

// 143. 重排链表
func reorderList(head *ListNode) {
	p, length := head, 0
	for p != nil {
		length += 1
		p = p.Next
	}

	//nodeSet := make(map[*ListNode]struct{})

	p = head
	for i := 0; i < (length+1)/2; i++ {

		q := p
		for j := 1; j < length-i*2; j++ {
			q = q.Next
		}

		if q == p {
			p.Next = nil
		} else if q == p.Next {
			q.Next = nil
		} else {
			temp := p.Next
			p.Next = q
			q.Next = temp
			p = temp
		}
	}
}

// 147. 对链表进行插入排序
func insertionSortList(head *ListNode) *ListNode {
	dummyNode := &ListNode{
		Val:  -5001,
		Next: nil,
	}

	for head != nil {
		val := head.Val
		p := dummyNode
		for p.Next != nil && val > p.Next.Val {
			p = p.Next
		}
		node := &ListNode{
			Val:  val,
			Next: p.Next,
		}
		p.Next = node

		head = head.Next
	}

	return dummyNode.Next
}
