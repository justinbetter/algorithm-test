package com.justinbetter.algorithhm;

public class dpProblem {

    public static void main(String[] args) {
        // write your code here
    }

    class maxProfitSolution {

        //多次交易需要记录上一次的卖出利润
        public int maxProfit2(int[] prices) {
            int n = prices.length;
            int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
            for (int i = 0; i < n; i++) {
                int temp = dp_i_0;
                dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
                dp_i_1 = Math.max(dp_i_1, temp - prices[i]);
            }
            return dp_i_0;
        }
        //-------------------------------------------------------------//

        //买卖股票的最佳时机
        //如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。
        //0表示卖出，1表示持有 情况下的利润；
        //注意：你不能在买入股票前卖出股票最大利润就是求卖出的时候最大的价格
        //循环结束后的d0就是最大利润
        public int maxProfit(int[] prices) {
            // 状态转移方程
//        dp[i][k][0] = Math.max(dp[i - 1][k][0], dp[i - 1][k][0] + prices[i]);
//        dp[i][k][1] = Math.max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i]);
            // k = 1
            int n = prices.length;
            int dp_i_0 = 0;
            int dp_i_1 = Integer.MIN_VALUE;
            for (int i = 0; i < n; i++) {
                dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
                dp_i_1 = Math.max(dp_i_1, -prices[i]);
            }
            return dp_i_0;
        }

        public int maxProfitOld(int prices[]) {
            int minprice = Integer.MAX_VALUE;
            int maxprofit = 0;
            for (int i = 0; i < prices.length; i++) {
                if (prices[i] < minprice)
                    minprice = prices[i];
                else if (prices[i] - minprice > maxprofit)
                    maxprofit = prices[i] - minprice;
            }
            return maxprofit;
        }

    }

}
