package com.justinbetter.algorithhm;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class dpProblem {

    public static void main(String[] args) {
        // write your code here
    }

    //俄罗斯套娃
    class maxEnvelopesSolution {

        public int maxEnvelopes(int[][] envelopes) {
            //按照w升序排列，h降序排列；逆序排序保证在 w 相同的数对中最多只选取一个 ，降低了信息量
            // sort on increasing in first dimension and decreasing in second
            Arrays.sort(envelopes, new Comparator<int[]>() {
                public int compare(int[] arr1, int[] arr2) {
                    if (arr1[0] == arr2[0]) {
                        return arr2[1] - arr1[1];
                    } else {
                        return arr1[0] - arr2[0];
                    }
                }
            });
            //将h遍历出来，获取h的最长递增子序列
            // extract the second dimension and run LIS
            int[] secondDim = new int[envelopes.length];
            for (int i = 0; i < envelopes.length; ++i) secondDim[i] = envelopes[i][1];
            return lengthOfLIS(secondDim);
        }

        //dp[i] 表示以 nums[i] 这个数结尾的最长递增子序列的长度
        //dp[i] = max(dp[i], dp[j] + 1) for j in [0, i)；只要之前的数都小于i就加入计算
        public int lengthOfLIS(int[] nums) {
            int[] dp = new int[nums.length];
            // base case：dp 数组全都初始化为 1
            Arrays.fill(dp, 1);
            for (int i = 0; i < nums.length; i++) {
                for (int j = 0; j < i; j++) {
                    if (nums[i] > nums[j])
                        dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }

            int res = 0;
            for (int i = 0; i < dp.length; i++) {
                res = Math.max(res, dp[i]);
            }
            return res;
        }

        //扑克牌的堆数
        public int lengthOfLISOld(int[] nums) {
            int[] dp = new int[nums.length];
            int len = 0;
            for (int num : nums) {
                //获取排序后的index
                int i = Arrays.binarySearch(dp, 0, len, num);
                //index为-1 说明没找到，i=0
                if (i < 0) {
                    i = -(i + 1);
                }
                //将dp0设置为num，
                dp[i] = num;
                if (i == len) {
                    len++;
                }
            }
            return len;
        }
    }

    //三角形最小路径和
    class minimumTotalSolution {
        public int minimumTotal(List<List<Integer>> triangle) {
            //dp[i][j] = min(dp[i+1][j+1],dp[i+1][j])+triangle[i][j], triangle[i][j] 表示位置 (i,j) 对应的元素值。
            int n = triangle.size();
            int[][] dp = new int[n + 1][n + 1];
            for (int i = n - 1; i >= 0; i--) { //底部行
                for (int j = 0; j <= i; j++) { //首列，三角形边界就是n
                    dp[i][j] = Math.min(dp[i + 1][j], dp[i + 1][j + 1]) + triangle.get(i).get(j);
                }
            }
            return dp[0][0];
        }

        public int minimumTotal2(List<List<Integer>> triangle) {
            //第一反应没写出来 dp[i] = Math.min(dp[i-1]+dp)
            //定义二维 dp 数组，将解法二中「自顶向下的递归」改为「自底向上的递推」。
            //dp[i][j] = min(dp[i+1][j+1],dp[i+1][j])+triangle[i][j], triangle[i][j] 表示位置 (i,j) 对应的元素值。
            int n = triangle.size();
            // dp[i][j] 表示从点 (i, j) 到底边的最小路径和。
            int[][] dp = new int[n + 1][n + 1];
            // 从三角形的最后一行开始递推。
            for (int i = n - 1; i >= 0; i--) {
                for (int j = 0; j <= i; j++) {
                    dp[i][j] = Math.min(dp[i + 1][j], dp[i + 1][j + 1]) + triangle.get(i).get(j);
                }
            }
            return dp[0][0];
        }
    }

    //最大子序和
    class maxSubArraySolution {

        public int maxSubArray(int[] nums) {
            int pre = 0;
            int ans = nums[0];
            for (int i = 0; i < nums.length; i++) {
                int currentNum = nums[i];
                pre = Math.max(pre + currentNum, currentNum);
                ans = Math.max(ans, pre);
            }
            return ans;
        }

        //动态规划，遍历每个位置的和下一个数的最大值即可
        public int maxSubArray2(int[] nums) {
            int pre = 0, maxAns = nums[0];
            for (int x : nums) {
                pre = Math.max(pre + x, x);
                maxAns = Math.max(maxAns, pre);
            }
            return maxAns;
        }

        public int maxSubArray1(int[] nums) {
            int ans = nums[0];
            int sum = 0;
            for (int num : nums) {
                if (sum > 0) {
                    sum += num;
                } else {
                    sum = num;
                }
                ans = Math.max(ans, sum);
            }
            return ans;
        }
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
