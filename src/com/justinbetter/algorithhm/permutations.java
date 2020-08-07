package com.justinbetter.algorithhm;

import java.util.ArrayList;
import java.util.List;

public class permutations {

    public static void main(String[] args) {
        // write your code here
        int[] nums = {1, 2, 3};
        System.out.println(permute(nums));
    }


    private static List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        //回溯： for 循环里面的递归，在递归调用之前「做选择」，在递归调用之后「撤销选择」。
        backtrack(res, nums, new ArrayList<Integer>(), new int[nums.length]);
        return res;

    }

    private static void backtrack(List<List<Integer>> res, int[] nums, ArrayList<Integer> tmp, int[] visited) {
        // 结束条件
        if (tmp.size() == nums.length) {
            //加入新数组
            res.add(new ArrayList<>(tmp));
            return;
        }
        // 循环+递归
        for (int i = 0; i < nums.length; i++) {
            if (visited[i] == 1) {
                continue;
            }
            visited[i] = 1;
            tmp.add(nums[i]);
            backtrack(res, nums, tmp, visited);
            visited[i] = 0;
            tmp.remove(tmp.size() - 1);
        }

    }

}
