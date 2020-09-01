package com.justinbetter.algorithhm;

import java.util.HashMap;
import java.util.Map;

public class btree {
    static Map<Integer, Integer> indexRootMap;

    public static void main(String[] args) {
        // write your code here
        System.out.println("test a code1");
        int[] preorder = {3, 9, 20, 15, 7};
        int[] inorder = {9, 3, 15, 20, 7};
        System.out.println("res" + buildTree(preorder, inorder));
    }

    static TreeNode buildTree(int[] preorder, int[] inorder) {
        // add 1，traverse left , traverse Right
        indexRootMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            indexRootMap.put(inorder[i], i);
        }
        return traverseBuild(preorder, inorder, 0, preorder.length - 1, 0, inorder.length - 1);
    }

    static TreeNode traverseBuild(int[] preorder, int[] inorder, int preLeft, int preRight, int inLeft, int inRight) {
        if (preLeft > preRight) {
            return null;
        }
        int rootValue = preorder[preLeft];
        TreeNode root = new TreeNode(preorder[preLeft]);
        //计算左子树节点数量
        int rootIndex = indexRootMap.get(rootValue);
        int nodeSize = rootIndex -  inLeft;
        root.left = traverseBuild(preorder, inorder, preLeft + 1, preLeft + nodeSize, inLeft, inLeft + nodeSize);
        root.right = traverseBuild(preorder, inorder, preLeft + nodeSize + 1, preRight, rootIndex + 1, inRight);
        return root;
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }

        @Override
        public String toString() {
            return super.toString();
        }
    }
}
