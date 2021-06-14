package com.justinbetter.algorithhm;

public class ListNodeProblem {
    static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    class isSubStructureSolution {
        public boolean isSubStructure(TreeNode A, TreeNode B) {
            return (A != null && B != null) && (dfs(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B));
        }

        boolean dfs(TreeNode A, TreeNode B) {
            if (B == null) return true;
            if (A == null || B.val != A.val) return false;
            return dfs(A.left, B.left) && dfs(A.right, B.right);
        }
    }

    //回文链表
    class isPalindromeSolution {

        public boolean isPalindrome(ListNode head) {
            //mine:反转右半边，和前一半一一比较
            //快慢指针 找到中点，注意奇数
            //注意起点别写错了，应该是一样的
            ListNode slow = head;
            ListNode fast = head;
            while (fast != null && fast.next != null) {
                slow = slow.next;
                fast = fast.next.next;
            }
            if (fast != null) { //说明是奇数
                slow = slow.next;
            }
            slow = reverse(slow);
            fast = head;
            //注意这里必须是slow != null
            while (slow != null) {
                if (fast.val != slow.val) {
                    return false;
                }
                fast = fast.next;
                slow = slow.next;
            }
            return true;
        }

        ListNode reverse(ListNode head) {
            if (head == null || head.next == null) return head;
            ListNode cur = reverse(head.next);
            head.next.next = head;
            head.next = null;
            return cur;
        }

    }

    static class Solution {

        ListNode mergeLR(ListNode left, ListNode right) {
            ListNode head = left;
            ListNode next = null;
            while (left.next != null) {
                next = right.next;
                right.next = left.next;
                left.next = right;
                left = right.next;
                right = next;
            }
            left.next = right;
            return head;
        }

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
        ListNode _node1 = new ListNode(12);
        ListNode _node2 = new ListNode(22);
        ListNode _node3 = new ListNode(32);
        ListNode _node4 = new ListNode(42);
        ListNode _node5 = new ListNode(52);
        _node4.next = _node5;
        _node3.next = _node4;
        _node2.next = _node3;
        _node1.next = _node2;
        Solution solution = new Solution();
        ListNode resNode = solution.mergeLR(node1, _node1);
        solution.printHead(resNode, "merge ");

    }


}
