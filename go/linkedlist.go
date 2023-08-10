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
