// 146. LRU 缓存
//函数 get 和 put 必须以 O(1) 的平均时间复杂度运行

package _go

type LRUCache struct {
	cap  int
	size int
	data map[int]*DLinkedNode
	head *DLinkedNode
	tail *DLinkedNode
}

type DLinkedNode struct {
	Key, Val   int
	prev, next *DLinkedNode
}

func Constructor(capacity int) LRUCache {
	return LRUCache{
		data: make(map[int]*DLinkedNode),
		cap:  capacity,
		size: 0,
		head: nil,
		tail: nil,
	}
}

func (this *LRUCache) up2Head(valNode *DLinkedNode) {
	if this.head == valNode {
		return
	}
	if valNode.prev != nil {
		valNode.prev.next = valNode.next
		if this.tail == valNode {
			this.tail = valNode.prev
		}
	}
	if valNode.next != nil {
		valNode.next.prev = valNode.prev
	}
	valNode.prev, valNode.next = nil, this.head
	if this.head != nil {
		this.head.prev = valNode
	}
	this.head = valNode
}

func (this *LRUCache) Get(key int) int {
	if valNode, ok := this.data[key]; ok && valNode != nil {
		this.up2Head(valNode)
		return valNode.Val
	} else {
		return -1
	}
}

func (this *LRUCache) Put(key int, value int) {
	if _, ok := this.data[key]; ok {
		this.data[key].Val = value
		this.up2Head(this.data[key])
		return
	}

	newNode := &DLinkedNode{
		Key:  key,
		Val:  value,
		prev: nil,
		next: this.head,
	}
	if this.head != nil {
		this.head.prev = newNode
	}
	if this.tail == nil {
		this.tail = newNode
	}
	this.head = newNode
	this.data[key] = newNode
	this.size++

	if this.size > this.cap {
		tail := this.tail
		delete(this.data, tail.Key)
		if tail.prev != nil {
			tail.prev.next = nil
		}
		this.tail = tail.prev
		tail = nil
		this.size--
	}
}
