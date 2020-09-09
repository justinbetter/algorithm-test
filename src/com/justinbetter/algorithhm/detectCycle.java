package com.justinbetter.algorithhm;

public class detectCycle {
    class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    public ListNode detectCycle(ListNode head) {
        //判断链表是否有️环，返回环入口
        //有环找到环中的第一个相遇点，根据第一个相遇点找到第二个相遇点就是环的入口
        ListNode slow = head;
        ListNode fast = head;
        while (true) {
            if (fast == null || fast.next == null) {
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                break;
            }
        }
        slow = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    public static void main(String[] args) {
        // write your code here
    }
}
