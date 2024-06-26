package top150

func hasCycle(head *ListNode) bool {
	p, q := head, head
	for p != nil && q != nil {
		if q.Next == nil {
			break
		}
		p, q = p.Next, q.Next.Next
		if p == q {
			return true
		}
	}

	return false
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	p, q, num := l1, l2, 0
	for p != nil && q != nil {
		if val := p.Val + q.Val + num; val >= 10 {
			p.Val, num = val-10, 1
		} else {
			p.Val, num = val, 0
		}

		//break
		if p.Next == nil || q.Next == nil {
			if p.Next == nil {
				p.Next = q.Next
			}
			for p.Next != nil && num > 0 {
				p = p.Next
				if val := p.Val + num; val >= 10 {
					p.Val, num = val-10, 1
				} else {
					p.Val, num = val, 0
				}
			}
			if num > 0 {
				p.Next = &ListNode{Val: num}
			}
			break
		}

		p, q = p.Next, q.Next
	}

	return l1
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	dummyNode := &ListNode{}
	k := dummyNode
	for list1 != nil && list2 != nil {
		if list1.Val < list2.Val {
			k.Next = list1
			list1 = list1.Next
		} else {
			k.Next = list2
			list2 = list2.Next
		}
		k = k.Next
	}
	if list1 == nil {
		k.Next = list2
	}
	if list2 == nil {
		k.Next = list1
	}

	return dummyNode.Next
}

func copyRandomList(head *Node) *Node {
	if head == nil {
		return nil
	}
	for p := head; p != nil; {
		next := p.Next
		p.Next = &Node{Val: p.Val, Next: next}
		p = next
	}
	for p := head; p != nil; {
		next := p.Next
		if p.Random != nil {
			next.Random = p.Random.Next
		}
		p = next.Next
	}
	newHead := head.Next
	for p := head; p != nil; {
		next := p.Next
		if next != nil {
			p.Next = next.Next
		}
		p = next
	}
	return newHead
}

func reverseBetween(head *ListNode, left int, right int) *ListNode {
	dummyNode := &ListNode{Next: head}
	prev := dummyNode
	for i := 1; i < left; i++ {
		prev = prev.Next
	}

	p, q := prev.Next, prev.Next.Next
	length := right - left
	for i := 0; i < length; i++ {
		temp := q.Next
		q.Next = p
		p = q
		q = temp
	}
	prev.Next.Next = q
	prev.Next = p

	return dummyNode.Next
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummyNode := &ListNode{Next: head}
	pre, p := dummyNode, head
	for i := 1; p.Next != nil; {
		if i < n {
			p = p.Next
			i++
			continue
		}
		pre, p = pre.Next, p.Next
	}
	pre.Next = pre.Next.Next
	return dummyNode.Next
}

func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	dummyNode := &ListNode{}
	p, q := dummyNode, head
	for t := q; q.Next != nil; {
		if q.Val != q.Next.Val {
			if t == q {
				p.Next = q
				p = q
				t = q.Next
				p.Next = nil
			} else {
				t = q.Next
			}
			q = t
		} else {
			q = q.Next
		}

		if q.Next == nil && t == q {
			p.Next = q
		}
	}

	return dummyNode.Next
}

func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k == 0 {
		return head
	}

	length := 1
	p := head
	for ; p.Next != nil; p = p.Next {
		length++
	}
	p.Next = head

	p = head
	for i := 1; i < length-k%length; p = p.Next {
		i++
	}
	head = p.Next
	p.Next = nil

	return head
}

func partition(head *ListNode, x int) *ListNode {
	dummyNode := &ListNode{}
	var tempHead *ListNode
	var tempPre *ListNode

	k, p := dummyNode, head
	for p != nil {
		next := p.Next
		if p.Val < x {
			k.Next, p.Next = p, nil
			k, p = k.Next, next
		} else {
			if tempHead == nil {
				tempHead, tempPre = p, p
			} else {
				tempPre.Next = p
				tempPre = tempPre.Next
			}
			tempPre.Next = nil
			p = next
		}
	}
	k.Next = tempHead
	return dummyNode.Next
}
