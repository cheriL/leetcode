package _go

// 283. 移动零
func moveZeroes(nums []int) {
	i, j := 0, 0
	for ; i < len(nums); i++ {
		if nums[i] != 0 {
			if i != j {
				nums[j] = nums[i]
			}
			j++
		}
	}

	for i = j; i < len(nums); i++ {
		nums[i] = 0
	}
}

// 206. 反转链表
func reverseList(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	p, q := head, head.Next
	p.Next = nil
	for q != nil {
		temp := q.Next
		q.Next = p
		p, q = q, temp
	}
	return p
}

// 25. K 个一组翻转链表
func reverseKGroup(head *ListNode, k int) *ListNode {
	if head == nil {
		return nil
	}

	var reverse func(*ListNode) *ListNode
	reverse = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}
		p := reverse(node.Next)
		p.Next = node
		node.Next = nil
		return node
	}

	var reverseK func(*ListNode) *ListNode
	reverseK = func(node *ListNode) *ListNode {
		p := node
		for i := 1; i < k; i++ {
			if p == nil {
				return node
			}
			p = p.Next
		}
		if p == nil {
			return node
		}
		q := p.Next
		p.Next = nil
		reverse(node)
		node.Next = reverseK(q)
		return p
	}

	return reverseK(head)
}

// 437. 路径总和 III
func pathSum3(root *TreeNode, targetSum int) int {
	count := 0

	var calTree func(*TreeNode, int, int)
	calTree = func(node *TreeNode, val int, target int) {
		if node == nil {
			return
		}
		sum := val + node.Val
		if sum == target {
			count++
		}
		calTree(node.Left, sum, target)
		calTree(node.Right, sum, target)
	}

	var traversal func(*TreeNode)
	traversal = func(node *TreeNode) {
		if node == nil {
			return
		}
		if node.Left != nil {
			traversal(node.Left)
		}
		if node.Right != nil {
			traversal(node.Right)
		}
		calTree(node, 0, targetSum)
	}
	traversal(root)
	return count
}

// 347. 前 K 个高频元素
func topKFrequent(nums []int, k int) []int {
	numCntMapping := map[int]int{}
	for i := 0; i < len(nums); i++ {
		if _, ok := numCntMapping[nums[i]]; ok {
			numCntMapping[nums[i]]++
		} else {
			numCntMapping[nums[i]] = 1
		}
	}
	var numCntList []interface{}
	for n, c := range numCntMapping {
		numCntList = append(numCntList, &struct {
			num int
			cnt int
		}{num: n, cnt: c})
	}
	numCntMapping = map[int]int{}

	var buildMaxHeap func([]interface{}, int)
	var maxHeapify func([]interface{}, int, int)
	buildMaxHeap = func(nc []interface{}, size int) {
		for i := size / 2; i >= 0; i-- {
			maxHeapify(nc, i, size)
		}
	}
	maxHeapify = func(nc []interface{}, idx int, size int) {
		l, r, maxIdx := idx*2+1, idx*2+2, idx
		if l < size && nc[l].(*struct {
			num int
			cnt int
		}).cnt > nc[maxIdx].(*struct {
			num int
			cnt int
		}).cnt {
			maxIdx = l
		}
		if r < size && nc[r].(*struct {
			num int
			cnt int
		}).cnt > nc[maxIdx].(*struct {
			num int
			cnt int
		}).cnt {
			maxIdx = r
		}
		if maxIdx != idx {
			nc[maxIdx], nc[idx] = nc[idx], nc[maxIdx]
			maxHeapify(nc, maxIdx, size)
		}
	}

	results := make([]int, 0, k)
	length := len(numCntList)
	buildMaxHeap(numCntList, length)
	for i := 0; i < k; i++ {
		results = append(results, numCntList[0].(*struct {
			num int
			cnt int
		}).num)
		numCntList[0] = numCntList[length-1]
		length--
		maxHeapify(numCntList, 0, length)
	}
	return results
}
