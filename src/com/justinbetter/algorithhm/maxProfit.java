package com.justinbetter.algorithhm;

public class maxProfit {

    public static void main(String[] args) {
        // write your code here
        int[] start = {7, 1, 5, 3, 6, 4};
        System.out.print(maxProfit_k_1(start));
    }

    public static int maxProfit_k_1(int[] prices) {
        // 状态转移方程
//        dp[i][k][0] = Math.max(dp[i - 1][k][0], dp[i - 1][k][0] + prices[i]);
//        dp[i][k][1] = Math.max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i]);
        // k = 1
        int n = prices.length;
        int dp_i_0 = 0;
        int dp_i_1 = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
            dp_i_1 = Math.max(dp_i_1,  - prices[i]);
        }
        return dp_i_0;
    }

}
