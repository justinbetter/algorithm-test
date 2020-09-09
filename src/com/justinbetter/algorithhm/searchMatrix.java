package com.justinbetter.algorithhm;

public class searchMatrix {

    public static void main(String[] args) {
        // write your code here
        searchMatrix solution = new searchMatrix();
        int[][] matrix = {{1, 2, 3}, {4, 5, 6}};
        System.out.println(solution.searchMatrix(matrix, 4));
    }

    boolean searchMatrix(int[][] matrix, int target) {
        // 先获取所在行数，再开始普通的二分查找
        return false;
    }
}
