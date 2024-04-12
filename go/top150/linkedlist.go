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
