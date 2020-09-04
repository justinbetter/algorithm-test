package com.justinbetter.algorithhm;

public class ListNode1 {
    static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    static class Solution {
        //链表1->2->3->4->5，按照1->5->2->4->3重新组装
        ListNode assemble(ListNode head) {
            //双指针获取中点，反转后半部分链表，拼接新链表
            ListNode resHead = new ListNode(0);
            ListNode initHead = resHead;
            ListNode fast = head.next;
            ListNode slow = head;
            while (fast != null && fast.next != null) {
                fast = fast.next.next;
                slow = slow.next;
            }
            ListNode mid = slow.next;
            printHead(head, "head");
            printHead(mid, "mid");
            ListNode lastNode = reverse(mid);
            printHead(lastNode, "lastNode");
            while (lastNode != null || head != null) {
                resHead.next = head;
                head = head.next;
                resHead = resHead.next;
                resHead.next = lastNode;
                if (lastNode == null) { //判断null 才跳出，不然中间的node没法读取
                    break;
                }
                resHead = resHead.next;
                lastNode = lastNode.next;
            }
            System.out.println();
            printHead(initHead.next, "resHead");
            return initHead.next;
        }

        void printHead(ListNode initHead, String prefix) {
            ListNode fake = initHead;
            while (fake != null) {
                System.out.print(prefix + fake.val + " ");
                fake = fake.next;
            }
        }

        ListNode reverse(ListNode head) {
            if (head.next == null) return head;
            ListNode last = reverse(head.next);
            head.next.next = head;
            head.next = null;
            return last;
        }
    }

    public static void main(String[] args) {
        // write your code here
        ListNode node1 = new ListNode(1);
        ListNode node2 = new ListNode(2);
        ListNode node3 = new ListNode(3);
        ListNode node4 = new ListNode(4);
        ListNode node5 = new ListNode(5);
        node4.next = node5;
        node3.next = node4;
        node2.next = node3;
        node1.next = node2;
        ListNode resNode = new Solution().assemble(node1);
    }


}
