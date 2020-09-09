package com.justinbetter.algorithhm;

public class sortList {

    public static void main(String[] args) {
        // 在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序。
    }
}

class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
    }
}

class Solution {
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        //递归归并排序
        //找到中点，递归中点分割，归并合并
        ListNode slow = head;
        ListNode fast = head.next;
        //fast走完 slow就是中点
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode center = slow.next;
        slow.next = null;
        ListNode left = sortList(head);
        ListNode right = sortList(center);
        ListNode tmp = new ListNode(0);
        ListNode res = tmp;
        while (left != null && right != null) {
            if (left.val < right.val) {
                tmp.next = left;
                left = left.next;
            } else {
                tmp.next = right;
                right = right.next;
            }
            tmp = tmp.next;
        }
        //添加尾点
        tmp.next = left == null ? right : left;
        //返回头部
        return res.next;


        //-----
        //从底至顶 归并排序
        //获取链表长度，根据每一步，循环断链合并，最后合并的链表超越链表长度即返回结果
    }
}
