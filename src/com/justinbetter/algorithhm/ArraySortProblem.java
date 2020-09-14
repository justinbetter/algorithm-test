package com.justinbetter.algorithhm;

import java.util.*;

public class ArraySortProblem {

    public static void main(String[] args) {
        // write your code here
    }
    //三数之和
    class threeSumSolution {
        public List<List<Integer>> threeSum(int[] nums) {
            List<List<Integer>> ans = new ArrayList();
            int len = nums.length;
            if (nums == null || len < 3) {
                return ans;
            }
            Arrays.sort(nums);
            for (int i = 0; i < len; i++) {
                if (nums[i] > 0) {
                    break;
                }
                if (i > 0 && nums[i] == nums[i-1]) continue;
                int L = i + 1;
                int R = len - 1;
                while (L < R) {
                    int sum = nums[i] + nums[L] + nums[R];
                    if (sum == 0) {
                        ans.add(Arrays.asList(nums[i],nums[L],nums[R]));
                        while (L < R && nums[L] == nums[L+1]) L++;
                        while (L < R && nums[R] == nums[R-1]) R--;
                        L++;
                        R--;
                    }else if (sum < 0) {
                        L++;
                    }else if (sum > 0){
                        R--;
                    }
                }
            }
            return ans;

        }
    }

}
