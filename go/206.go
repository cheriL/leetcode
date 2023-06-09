package _go

/**206
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {
	var pre *ListNode
	cur := head
	if cur == nil {
		return cur
	}
	for cur.Next != nil {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	cur.Next = pre
	return cur
}
