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
