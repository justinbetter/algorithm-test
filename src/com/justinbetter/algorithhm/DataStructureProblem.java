package com.justinbetter.algorithhm;

import java.util.*;

public class DataStructureProblem {

    public static void main(String[] args) {
        // write your code here
    }

    //LFUCache
    class LFUCache {

        public LFUCache(int capacity) {

        }

        public int get(int key) {
            return -1;
        }

        public void put(int key, int value) {

        }
    }

    //O(1)数据结构
    class AllOne {
        class DListNode {
            int value;
            HashSet<String> keySet;
            DListNode pre;
            DListNode next;

            DListNode() {
                keySet = new HashSet<>();
            }
        }

        HashMap<String, Integer> keyMap = new HashMap<>();
        HashMap<Integer, DListNode> valueMap = new HashMap<>();
        DListNode head, tail;

        /**
         * Initialize your data structure here.
         */
        public AllOne() {
            head = new DListNode();
            tail = new DListNode();
            head.next = tail;
            tail.pre = head;
        }

        /**
         * Insert new node in the order list.
         */
        public void insertNode(String key, int curvalue, int dir) {
            DListNode newnode = new DListNode();
            valueMap.put(curvalue + dir, newnode);
            newnode.keySet.add(key);
            if (curvalue == 0) {
                newnode.next = head.next;
                head.next.pre = newnode;
                newnode.pre = head;
                head.next = newnode;
                return;
            }

            DListNode curnode = valueMap.get(curvalue);
            if (dir == 1) {
                newnode.next = curnode.next;
                curnode.next.pre = newnode;
                newnode.pre = curnode;
                curnode.next = newnode;
            } else if (dir == -1) {
                newnode.pre = curnode.pre;
                newnode.next = curnode;
                curnode.pre.next = newnode;
                curnode.pre = newnode;
            }
        }

        /**
         * Remove the empty node of the order list.
         */
        public void removeNode(int curvalue) {
            DListNode curnode = valueMap.get(curvalue);
            if (curnode.keySet.isEmpty()) {
                valueMap.remove(curvalue);
                curnode.next.pre = curnode.pre;
                curnode.pre.next = curnode.next;
                curnode.next = null;
                curnode.pre = null;
            }
        }

        /**
         * Inserts a new key <Key> with value 1. Or increments an existing key by 1.
         */
        public void inc(String key) {
            if (keyMap.containsKey(key)) {
                int curvalue = keyMap.get(key);
                valueMap.get(curvalue).keySet.remove(key);
                keyMap.replace(key, curvalue + 1);
                if (valueMap.containsKey(curvalue + 1)) {
                    valueMap.get(curvalue + 1).keySet.add(key);
                } else {
                    insertNode(key, curvalue, 1);
                }
                removeNode(curvalue);
            } else {
                keyMap.put(key, 1);
                if (valueMap.containsKey(1)) {
                    valueMap.get(1).keySet.add(key);
                } else {
                    insertNode(key, 0, 1);
                }
            }
        }

        /**
         * Decrements an existing key by 1. If Key's value is 1, remove it from the data structure.
         */
        public void dec(String key) {
            if (keyMap.containsKey(key)) {
                int curvalue = keyMap.get(key);
                DListNode curnode = valueMap.get(curvalue);
                curnode.keySet.remove(key);
                if (curvalue == 1) {
                    keyMap.remove(key);
                    removeNode(curvalue);
                    return;
                }
                keyMap.replace(key, curvalue - 1);
                if (valueMap.containsKey(curvalue - 1)) {
                    valueMap.get(curvalue - 1).keySet.add(key);
                } else {
                    insertNode(key, curvalue, -1);
                }
                removeNode(curvalue);
            }
        }

        /**
         * Returns one of the keys with maximal value.
         */
        public String getMaxKey() {
            return (tail.pre == head) ? "" : tail.pre.keySet.iterator().next();
        }

        /**
         * Returns one of the keys with Minimal value.
         */
        public String getMinKey() {
            return (head.next == tail) ? "" : head.next.keySet.iterator().next();
        }
    }

    //哈希表+双端链表
    public class LRUCache {
        class DLinkedNode {
            int key;
            int value;
            DLinkedNode prev;
            DLinkedNode next;

            public DLinkedNode() {
            }

            public DLinkedNode(int _key, int _value) {
                key = _key;
                value = _value;
            }
        }

        private Map<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
        private int size;
        private int capacity;
        private DLinkedNode head, tail;

        public LRUCache(int capacity) {
            this.size = 0;
            this.capacity = capacity;
            // 使用伪头部和伪尾部节点
            head = new DLinkedNode();
            tail = new DLinkedNode();
            head.next = tail;
            tail.prev = head;
        }

        public int get(int key) {
            DLinkedNode node = cache.get(key);
            if (node == null) {
                return -1;
            }
            // 如果 key 存在，先通过哈希表定位，再移到头部
            moveToHead(node);
            return node.value;
        }

        public void put(int key, int value) {
            DLinkedNode node = cache.get(key);
            if (node == null) {
                // 如果 key 不存在，创建一个新的节点
                DLinkedNode newNode = new DLinkedNode(key, value);
                // 添加进哈希表
                cache.put(key, newNode);
                // 添加至双向链表的头部
                addToHead(newNode);
                ++size;
                if (size > capacity) {
                    // 如果超出容量，删除双向链表的尾部节点
                    DLinkedNode tail = removeTail();
                    // 删除哈希表中对应的项
                    cache.remove(tail.key);
                    --size;
                }
            } else {
                // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
                node.value = value;
                moveToHead(node);
            }
        }

        private void addToHead(DLinkedNode node) {
            //当前节点前插入头部
            node.prev = head;
            //当前节点后插入原头部后节点
            node.next = head.next;
            //原头部后节点前连接当前节点
            head.next.prev = node;
            //头部连接当前节点
            head.next = node;
        }

        private void removeNode(DLinkedNode node) {
            //当前节点前节点连接当前节点后的节点
            node.prev.next = node.next;
            //当前节点后节点前连接当前节点前的节点
            node.next.prev = node.prev;
        }

        private void moveToHead(DLinkedNode node) {
            removeNode(node);
            addToHead(node);
        }

        private DLinkedNode removeTail() {
            DLinkedNode res = tail.prev;
            removeNode(res);
            return res;
        }
    }

    //最小栈
    class MinStack {
        Deque<Integer> xStack;
        Deque<Integer> minStack;

        public MinStack() {
            xStack = new LinkedList<Integer>();
            minStack = new LinkedList<Integer>();
            minStack.push(Integer.MAX_VALUE);
        }

        public void push(int x) {
            xStack.push(x);
            minStack.push(Math.min(minStack.peek(), x));
        }

        public void pop() {
            xStack.pop();
            minStack.pop();
        }

        public int top() {
            return xStack.peek();
        }

        public int getMin() {
            return minStack.peek();
        }
    }


}
