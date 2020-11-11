package com.justinbetter.algorithhm;

import java.util.*;

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

    //LC105 从前序中序遍历构造二叉树
    class buildTreeSolution {

        Map<Integer, Integer> indexRootMap;

        public TreeNode buildTree(int[] preorder, int[] inorder) {
            // add 1，traverse left , traverse Right
            //通过前序遍历确定根节点，中序遍历确定左右紫薯
            HashMap<Integer, Integer> map = new HashMap<>();
            for (int i = 0; i < inorder.length; i++) {
                map.put(inorder[i], i);
            }
            return buildSubTree(map, preorder, inorder, 0, preorder.length - 1, 0, inorder.length - 1);
        }

        private TreeNode buildSubTree(HashMap<Integer, Integer> map, int[] preorder, int[] inorder, int preLeft, int preRight, int inLeft, int inRight) {
            //终止条件
            if (preLeft > preRight || inLeft > inRight) {
                return null;
            }
            //获取根节点
            int rootValue = preorder[preLeft];
            TreeNode root = new TreeNode(rootValue);
            //获取左右节点
            int indexInOrder = map.get(rootValue);
            int nodeSize = indexInOrder - inLeft;
            root.left = buildSubTree(map, preorder, inorder, preLeft + 1, preLeft + nodeSize, inLeft, inLeft + nodeSize);
            root.right = buildSubTree(map, preorder, inorder, preLeft + nodeSize + 1, preRight, indexInOrder + 1, inRight);
            return root;
        }


        TreeNode traverseBuild(int[] preorder, int[] inorder, int preLeft, int preRight, int inLeft, int inRight) {
            if (preLeft > preRight) {
                return null;
            }
            int rootValue = preorder[preLeft];
            TreeNode root = new TreeNode(preorder[preLeft]);
            //计算左子树节点数量
            int rootIndex = indexRootMap.get(rootValue);
            int nodeSize = rootIndex - inLeft;
            root.left = traverseBuild(preorder, inorder, preLeft + 1, preLeft + nodeSize, inLeft, inLeft + nodeSize);
            root.right = traverseBuild(preorder, inorder, preLeft + nodeSize + 1, preRight, rootIndex + 1, inRight);
            return root;
        }
    }

    //LC94 二叉树中序遍历
    class inorderTraversalSolution {
        public List<Integer> inorderTraversal(TreeNode root) {
            //遍历根节点 左节点 右节点
            List<Integer> res = new ArrayList<>();
            traverse(root, res);
            return res;
        }

        private void traverse(TreeNode root, List<Integer> res) {
            if (root == null) return;
            traverse(root.left, res);
            res.add(root.val);
            traverse(root.right, res);

        }

        public List<Integer> inorderTraversal2(TreeNode root) {
            //迭代遍历
            List<Integer> res = new ArrayList<>();
            Deque<TreeNode> stack = new LinkedList<>();
            while (root != null || !stack.isEmpty()) {
                //压栈
                while (root != null) {
                    stack.push(root);
                    root = root.left;
                }
                root = stack.pop();
                res.add(root.val);
                root = root.right;
            }
            return res;
        }
    }

    class HJSolution {
        public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
            //思路：
            //新建两个辅助node，保存倒序排列的l1 l2，不需要；因为题目本身就是逆序的
            //循环两个node 直到最后一个节点
            ListNode left = l1;
            ListNode right = l2;
            int carry = 0;
            ListNode tmp = new ListNode(0);
            ListNode res = tmp;
            while (left != null || right != null) {
                int x = (left == null) ? 0 : left.val;
                int y = (right == null) ? 0 : right.val;
                int sum = x + y + carry;
                carry = sum / 10;
                ListNode now = new ListNode(sum % 10);
                res.next = now;
                res = res.next;
                if (left != null) {
                    left = left.next;
                }
                if (right != null) {
                    right = right.next;
                }
            }
            if (carry > 0) {
                res.next = new ListNode(carry);
            }
            return tmp.next;


        }

        public List<Integer> preorderTraversal(TreeNode root) {
            //stack
            LinkedList<TreeNode> stack = new LinkedList<>();
            LinkedList<Integer> ans = new LinkedList<>();
            if (root == null) {
                return ans;
            }
            stack.push(root);
            while (!stack.isEmpty()) {
                TreeNode node = stack.poll();
                ans.add(node.val);
                if (node.right != null) {
                    stack.push(node.right);
                }
                if (node.left != null) {
                    stack.push(node.left);
                }
            }
            return ans;

        }

        public List<Integer> _preorderTraversal(TreeNode root) {
            LinkedList<TreeNode> stack = new LinkedList<>();
            LinkedList<Integer> output = new LinkedList<>();
            if (root == null) {
                return output;
            }

            stack.add(root);
            while (!stack.isEmpty()) {
                TreeNode node = stack.pollLast();
                output.add(node.val);
                if (node.right != null) {
                    stack.add(node.right);
                }
                if (node.left != null) {
                    stack.add(node.left);
                }
            }
            return output;
        }

        List<Integer> treeFirstSearch(TreeNode root) {
            List<Integer> ans = new LinkedList<>();
            if (root != null) {
                ans.add(root.val);
            }
            TreeNode left = root.left;
            TreeNode right = root.right;
            while (left != null) {
                ans.add(left.val);
                left = left.left;
            }

            while (root.right != null) {

            }
            return ans;
        }

    }

    //二叉树的最大长度
    class diameterOfBinaryTreeSolution {
        //dfs
        //递归节点深度，如果节点为null 说明到底了 返回深度0
        //深度max L R, + 1根节点
        //遍历的节点数-1 就是路径的长度
        int ans = 0;

        public int diameterOfBinaryTree(TreeNode root) {
            //深度遍历，遍历的节点数-1就是最大长度
            ans = 1;
            depth(root);
            return ans - 1;
        }

        int depth(TreeNode node) {
            if (node == null) {
                return 0;
            }
            int L = depth(node.left);
            int R = depth(node.right);
            ans = Math.max(ans, L + R + 1);
            return Math.max(L, R) + 1;
        }
    }

    //验证二叉树是否是bst
    class isValidBSTSolution {
        public boolean isValidBST(TreeNode root) {
            return check(root, null, null);
        }

        public boolean check(TreeNode root, TreeNode min, TreeNode max) {
            // 空树认为是BST
            if (root == null) {
                return true;
            }
            // 一次只检查一个节点，看这个节点是否破坏了BST特性
            if (min != null && root.val <= min.val) {
                return false;
            }
            if (max != null && root.val >= max.val) {
                return false;
            }
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
