package com.justinbetter.algorithhm;

public class BTreeProblem {
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    public static void main(String[] args) {
        // write your code here
    }

    //验证二叉树是否是bst
    class isValidBSTSolution {
        public boolean isValidBST(TreeNode root) {
            return check(root, null, null);
        }

        public boolean check(TreeNode root, TreeNode min, TreeNode max) {
            // 空树认为是BST
            if (root == null) { return true; }
            // 一次只检查一个节点，看这个节点是否破坏了BST特性
            if (min != null && root.val <= min.val) { return false; }
            if (max != null && root.val >= max.val) { return false; }
            // 对于左子树的所有节点值来说，最小值为min，最大值为root
            // 对于右子树的所有节点值来说，最小值为root，最大值为max
            return check(root.left, min, root) && check(root.right, root, max);
        }

        public boolean isValidBST2(TreeNode root) {
            if (root == null) return true;
            if (root.left != null && root.val > root.left.val) {
                return isValidBST(root.left);
            }
            if (root.right != null && root.val < root.right.val) {
                return isValidBST(root.right);
            }
            return false;
        }

        public boolean isValid(TreeNode root) {
            if (root == null) {
                return true;
            }
            if (root.left != null) {
                if (root.val > root.left.val) {
                    return isValid(root.left);
                } else {
                    return false;
                }
            }

            if (root.right != null) {
                if (root.val < root.right.val) {
                    return isValid(root.right);
                } else {
                    return false;
                }
            }
            return true;
        }
    }

    //二叉树后序遍历 从下往上
    //递归判断
    TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // base case
        if (root == null) return null;
        if (root == p || root == q) return root;


        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        // 情况 1
        if (left != null && right != null) {
            return root;
        }
        // 情况 2
        if (left == null && right == null) {
            return null;
        }
        // 情况 3
        return left == null ? right : left;
    }
}
