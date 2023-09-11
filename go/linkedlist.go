// Package _go https://leetcode.cn/tag/linked-list/problemset/
package _go

import (
	"math"
	"math/rand"
)

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

// 148. 排序链表
func sortList(head *ListNode) *ListNode {
	mergeFunc := func(left *ListNode, right *ListNode) *ListNode {
		dummyNode := &ListNode{}
		p, q, k := left, right, dummyNode
		for {
			if p == nil {
				k.Next = q
				break
			}
			if q == nil {
				k.Next = p
				break
			}
			if p.Val < q.Val {
				node := &ListNode{Val: p.Val}
				k.Next = node
				k = k.Next
				p = p.Next
			} else {
				node := &ListNode{Val: q.Val}
				k.Next = node
				k = k.Next
				q = q.Next
			}
		}
		return dummyNode.Next
	}

	var mergeSort func(*ListNode) *ListNode
	mergeSort = func(head *ListNode) *ListNode {
		if head == nil || head.Next == nil {
			return head
		}

		p, length := head, 0
		for p != nil {
			length++
			p = p.Next
		}
		mid := head
		for i := 1; i < length/2; i++ {
			mid = mid.Next
		}

		rightList := mergeSort(mid.Next)
		mid.Next = nil
		leftList := mergeSort(head)
		return mergeFunc(leftList, rightList)
	}

	return mergeSort(head)
}

// 160. 相交链表
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	p, q := headA, headB
	for p != q {
		p, q = p.Next, q.Next
		if p == nil && q == nil {
			return nil
		}

		if p == nil {
			p = headB
		}
		if q == nil {
			q = headA
		}
	}
	return p
}

// 203. 移除链表元素
func removeElements(head *ListNode, val int) *ListNode {
	dummyNode := &ListNode{Next: head}
	p, q := dummyNode, dummyNode.Next
	for q != nil {
		if q.Val == val {
			p.Next = q.Next
			q = nil
			q = p.Next
		} else {
			p = q
			q = q.Next
		}
	}
	return dummyNode.Next
}

// 234. 回文链表
func isPalindrome1(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return true
	}
	dummyNode := &ListNode{Next: head}
	s, f := dummyNode, dummyNode
	for f != nil && f.Next != nil {
		s = s.Next
		f = f.Next.Next
	}

	p, q := s, s.Next
	for q != nil {
		next := q.Next
		q.Next = p
		p = q
		q = next
	}
	q = p
	p = head
	for {
		if p == q {
			break
		}

		if p.Val != q.Val {
			return false
		}

		if p.Next == q && q.Next == p {
			break
		}

		p = p.Next
		q = q.Next
	}
	return true
}

// 237. 删除链表中的节点
func deleteNode(node *ListNode) {
	if node != nil {
		if node.Next != nil {
			node.Val = node.Next.Val
			temp := node.Next.Next
			node.Next = nil
			node.Next = temp
		}
	}
}

// 328. 奇偶链表
// 你必须在 O(1) 的额外空间复杂度和 O(n) 的时间复杂度下解决这个问题
func oddEvenList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	dummyNode := &ListNode{}
	t := dummyNode
	i, p, q := 0, head, head.Next
	for q != nil {
		if i%2 == 0 {
			t.Next = q
			q = q.Next
			p.Next = q
			t = t.Next
			t.Next = nil
		} else {
			q = q.Next
			p = p.Next
		}
		i++
	}
	p.Next = dummyNode.Next
	return head
}

// 445. 两数相加 II
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	//转换成数字再相加会溢出
	//反转链表想加，或栈
	if l1 == nil && l2 == nil {
		return &ListNode{Val: 0}
	}

	var stack1 []int
	var stack2 []int
	push := func(stack *[]int, val int) {
		*stack = append(*stack, val)
	}
	pop := func(stack *[]int) int {
		val := (*stack)[len(*stack)-1]
		*stack = (*stack)[:len(*stack)-1]
		return val
	}

	for ; l1 != nil; l1 = l1.Next {
		push(&stack1, l1.Val)
	}
	for ; l2 != nil; l2 = l2.Next {
		push(&stack2, l2.Val)
	}
	dummyNode := &ListNode{}
	appendNode := func(val int) {
		node := &ListNode{
			Val:  val,
			Next: dummyNode.Next,
		}
		dummyNode.Next = node
	}
	flag := 0
	for {
		x, y := pop(&stack1), pop(&stack2)
		sum := x + y + flag
		val := sum % 10
		flag = sum / 10
		appendNode(val)
		if len(stack1) == 0 {
			for len(stack2) > 0 {
				sum := pop(&stack2) + flag
				val := sum % 10
				flag = sum / 10
				appendNode(val)
			}
			break
		}
		if len(stack2) == 0 {
			for len(stack1) > 0 {
				sum := pop(&stack1) + flag
				val := sum % 10
				flag = sum / 10
				appendNode(val)
			}
			break
		}
	}
	if flag > 0 {
		appendNode(flag)
	}

	return dummyNode.Next
}

// 382.
func GetRandom(head *ListNode) int {
	res := 0
	for p, i := head, 1; p != nil; p = p.Next {
		//[0, n)
		val := rand.Intn(i)
		if val == 0 {
			res = p.Val
		}
		i++
	}
	return res
}

// 876. 链表的中间结点
func middleNode(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	p, q := head, head.Next.Next
	for q != nil && q.Next != nil {
		p = p.Next
		q = q.Next.Next
	}
	return p.Next
}

// 1019. 链表中的下一个更大节点
func nextLargerNodes(head *ListNode) []int {
	var results []int
	p := head
	for p != nil {
		res, q := 0, p.Next
		for q != nil {
			if q.Val > p.Val {
				res = q.Val
				break
			}
			q = q.Next
		}
		p = p.Next
		results = append(results, res)
	}
	return results
}

// 1171. 从链表中删去总和值为零的连续节点
func removeZeroSumSublists(head *ListNode) *ListNode {
	pre, p := head, head
	for p != nil {
		q := p.Next
		sum := p.Val
		if sum == 0 {
			if head == p {
				head = p.Next
			}
			pre.Next = p.Next
			p = p.Next
			continue
		}
		for q != nil {
			sum += q.Val
			if sum == 0 {
				break
			}
			q = q.Next
		}
		if q != nil {
			if head == p {
				head, pre, p = q.Next, q.Next, q.Next
			} else {
				pre.Next = q.Next
			}
			p = q.Next
		} else {
			pre = p
			p = p.Next
		}
	}
	return head
}

// 1290. 二进制链表转整数
func getDecimalValue(head *ListNode) int {
	num := 0
	for head != nil {
		num = num*2 + head.Val
		head = head.Next
	}
	return num
}

// 1367. 二叉树中的链表
func isSubPath(head *ListNode, root *TreeNode) bool {
	var validateTree func(*TreeNode, *ListNode) bool
	var validateOne func(*TreeNode, *ListNode) bool

	validateOne = func(treeNode *TreeNode, listNode *ListNode) bool {
		if listNode == nil {
			return true
		}
		if treeNode == nil {
			return false
		}
		if treeNode.Val != listNode.Val {
			return false
		}
		return validateOne(treeNode.Left, listNode.Next) || validateOne(treeNode.Right, listNode.Next)
	}

	validateTree = func(treeNode *TreeNode, listNode *ListNode) bool {
		if listNode == nil {
			return true
		}
		if treeNode == nil {
			return false
		}

		if validateOne(treeNode, listNode) {
			return true
		}

		return validateTree(treeNode.Left, listNode) || validateTree(treeNode.Right, listNode)
	}

	return validateTree(root, head)
}

// 1669. 合并两个链表 1 <= a <= b < list1.length - 1
func mergeInBetween(list1 *ListNode, a int, b int, list2 *ListNode) *ListNode {
	if list1 == nil || list2 == nil {
		return list1
	}
	tail2 := list2
	for ; tail2.Next != nil; tail2 = tail2.Next {
	}

	p, q, i := list1, list1.Next, 1
	for q != nil {
		if i == a {
			p.Next = list2
		}
		if i == b {
			tail2.Next = q.Next
			break
		}
		p = q
		q = q.Next
		i++
	}
	return list1
}

// 1721. 交换链表中的节点
func swapNodes(head *ListNode, k int) *ListNode {
	if head == nil {
		return nil
	}

	var resHead *ListNode
	p, length := head, 0
	for p != nil {
		length++
		node := &ListNode{
			Val:  p.Val,
			Next: resHead,
		}
		resHead = node
		p = p.Next
	}

	p = head
	q := resHead
	for i := 1; p != nil; i++ {
		if i == k || i == length-k+1 {
			p.Val = q.Val
		}
		p = p.Next
		q = q.Next
	}
	return head
}

// 2095. 删除链表的中间节点
func deleteMiddle(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return nil
	}

	length := 0
	for p := head; p != nil; p = p.Next {
		length++
	}
	midIdx := length / 2
	p, q := head, head.Next
	for i := 1; i < midIdx; i++ {
		p = q
		q = q.Next
	}

	if q != nil {
		p.Next = q.Next
	} else {
		p.Next = nil
	}
	q = nil

	return head
}

// 2130. 链表最大孪生和
func pairSum(head *ListNode) int {
	var head2 *ListNode
	length := 0
	for p := head; p != nil; p = p.Next {
		length++
		node := &ListNode{
			Val:  p.Val,
			Next: head2,
		}
		head2 = node
	}

	max := 0
	for p, q, i := head, head2, 0; i <= length/2; p, q, i = p.Next, q.Next, i+1 {
		if sum := p.Val + q.Val; sum > max {
			max = sum
		}
	}

	return max
}

// 2181. 合并零之间的节点
func mergeNodes(head *ListNode) *ListNode {
	for pre, p := head, head; p != nil; pre, p = p, p.Next {
		if p.Val != 0 {
			continue
		}

		sum := 0
		q := p.Next
		for ; q != nil; q = q.Next {
			sum += q.Val
			if q.Val == 0 {
				break
			}
		}
		if q != nil {
			p.Val = sum
			p.Next = q
		} else {
			pre.Next = nil
			break
		}
	}
	return head
}

// 2058. 找出临界点之间的最小和最大距离
func nodesBetweenCriticalPoints(head *ListNode) []int {
	if head == nil || head.Next == nil {
		return []int{-1, -1}
	}
	pre, p, first := head, head.Next, -1
	minVal, maxVal := math.MaxInt32, -1
	for i, last := 1, -1; p.Next != nil; pre, p = pre.Next, p.Next {
		if pre.Val < p.Val && p.Next.Val < p.Val || pre.Val > p.Val && p.Next.Val > p.Val {
			if first == -1 {
				first = i
			} else {
				maxVal = i - first
			}
			if last != -1 {
				if minVal > i-last {
					minVal = i - last
					last = i
				}
			}
			last = i
		}

		i++
	}

	if first == -1 || maxVal == -1 {
		minVal = -1
	}
	return []int{minVal, maxVal}
}

// 2074. 反转偶数长度组的节点
func reverseEvenLengthGroups(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	pre, idx := head, head.Next
	//i表示第i组
	i := 2
	for idx != nil {
		//计算长度
		length := 0
		for p, j := idx, math.Abs(float64(i)); p != nil && float64(length) < j; p, length = p.Next, length+1 {
		}

		if i > 0 {
			if length%2 == 0 {
				p, q, j := idx, idx.Next, 1
				p.Next = nil
				for ; j < i && p != nil && q != nil; j++ {
					temp := q.Next
					q.Next = p
					p = q
					q = temp
				}
				pre.Next.Next = q
				temp := pre.Next
				pre.Next = p
				pre, idx = temp, q
				i++
			}
		} else {
			if length%2 != 0 {
				j := -1
				p, q := pre, idx
				for q != nil && j >= i {
					p, q = p.Next, q.Next
					j--
				}
				pre, idx = p, q
				i--
			}
		}
		i = -1 * i
	}
	return head
}

// 2487. 从链表中移除节点
func removeNodes(head *ListNode) *ListNode {
	var removeFn func(node *ListNode) *ListNode
	removeFn = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}

		node.Next = removeFn(node.Next)
		if node.Next.Val > node.Val {
			return node.Next
		}
		return node
	}
	return removeFn(head)
}

// 2807. 在链表中插入最大公约数
func insertGreatestCommonDivisors(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	greatestFn := func(a, b int) int {
		if a == b {
			return a
		}

		temp, min := a-b, b
		if a < b {
			temp, min = b-a, a
		}
		for temp != min {
			if temp > min {
				temp = temp - min
			} else {
				temp, min = min-temp, temp
			}
		}
		return temp
	}

	p, q := head, head.Next
	for q != nil {
		val := greatestFn(p.Val, q.Val)
		p.Next = &ListNode{Val: val, Next: q}
		p = q
		q = q.Next
	}
	return head
}
