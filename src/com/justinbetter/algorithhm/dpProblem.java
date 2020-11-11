package com.justinbetter.algorithhm;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class dpProblem {

    public static void main(String[] args) {
        // write your code here
        char[][] grid = new char[][]{{'1', '1', '1', '1', '0'}, {'1', '1', '0', '1', '0'}, {'1', '1', '0', '0', '0'}, {'0', '0', '0', '0', '0'}};
        System.out.println(new numIslandsSolution().numIslands(grid));
    }

    static class numIslandsSolution {
        boolean[][] visited;
        int m;
        int n;
        int[][] direct = new int[][]{{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
        char[][] grid;
        private static final int[][] DIRECTIONS = {{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
        private int rows;
        private int cols;

        public int numIslands(char[][] grid) {
            //回溯 dfs visited direction
            m = grid.length;
            n = grid[0].length;
//            visited = new boolean[m][n];
//            int res = 0;
//            grid = grid;
//            if (m == 0) {
//                return 0;
//            }
            rows = grid.length;
            cols = grid[0].length;

            this.grid = grid;
            visited = new boolean[m][n];
            int res = 0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    // 如果是岛屿中的一个点，并且没有被访问过，就进行深度优先遍历
                    if (!visited[i][j] && grid[i][j] == '1') {
                        dfs(i, j);
                        res++;
                    }
                }
            }
            return res;
        }

        public int numIslands2(char[][] grid) {
            rows = grid.length;
            if (rows == 0) {
                return 0;
            }
            cols = grid[0].length;

            this.grid = grid;
            visited = new boolean[rows][cols];
            int count = 0;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    // 如果是岛屿中的一个点，并且没有被访问过，就进行深度优先遍历
                    if (!visited[i][j] && grid[i][j] == '1') {
                        dfs(i, j);
                        count++;
                    }
                }
            }
            return count;
        }

        /**
         * 从坐标为 (i, j) 的点开始进行深度优先遍历
         *
         * @param i
         * @param j
         */
        private void dfs(int i, int j) {
            visited[i][j] = true;
            for (int k = 0; k < 4; k++) {
                int newX = i + DIRECTIONS[k][0];
                int newY = j + DIRECTIONS[k][1];
                // 如果不越界、还是陆地、没有被访问过
                if (inArea(newX, newY) && grid[newX][newY] == '1' && !visited[newX][newY]) {
                    dfs(newX, newY);
                }
            }
        }

        /**
         * 封装成 inArea 方法语义更清晰
         *
         * @param x
         * @param y
         * @return
         */
        private boolean inArea(int x, int y) {
            return x >= 0 && x < m && y >= 0 && y < n;
        }

//        private void dfsold(int i, int j, char[][] grid) {
//            visited[i][j] = true;
//            //开始移动
//            for (int k = 0; k < 4; k++) {
//                int x = i + direct[k][0];
//                int y = j + direct[k][1];
//                if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y] == '1' && !visited[x][y]) {
//                    dfs(x, y, grid);
//                }
//            }
//
//        }

        private void dfs2(int i, int j) {
            visited[i][j] = true;
            for (int k = 0; k < 4; k++) {
                int newX = i + direct[k][0];
                int newY = j + direct[k][1];
                // 如果不越界、还是陆地、没有被访问过
                if (inArea(newX, newY) && grid[newX][newY] == '1' && !visited[newX][newY]) {
                    dfs2(newX, newY);
                }
            }
        }


    }

    //不相交的握手
    static class numberOfWaysSolution {
        int[] memo; // 记忆数组

        public int numberOfWays2(int num_people) {
            memo = new int[1 + num_people]; // 初始化记忆数组
            return (int) help(num_people); // 递归求解
        }

        long help(int num_people) {
            // 如果记忆数组存在值，直接发挥
            if (memo[num_people] > 0) {
                return memo[num_people];
            }
            // 如果总人数小于等于2，返回1
            if (num_people <= 2) return 1;
            // 返回结果
            long res = 0;
            // 从相邻人开始循环可能握手的人（相隔人数为偶数）
            for (int i = 1; i < num_people; i += 2) {
                // 握手后将人分为两部分，两部分递归求解的乘积加入返回结果
                res += (help(i - 1) * help(num_people - i - 1)) % 1000000007;
            }
            // 将结果存入记忆数组
            memo[num_people] = (int) (res % 1000000007);
            return memo[num_people];
        }

        public int numberOfWays(int num_people) {
            if (num_people == 2) return 1;
            int mod = (int) (Math.pow(10, 9) + 7);
            // dp[i] 表示 i 个人的不会相交的握手方案数
            long[] dp = new long[num_people + 1];
            dp[0] = dp[1] = 0;  // 一个人或者没有人无法握手
            dp[2] = 1;  // 两个人只有一种握手
            //从i表示人数，遍历不同人数的握手方案
            for (int i = 3; i <= num_people; i++) {
                long sum = 0;
                //表示和第k个人握手后的握手方案，
                for (int k = 4; k <= i - 2; k += 2) {
                    sum += (dp[k - 2] * dp[i - k]) % mod;
                }
                dp[i] = (2 * dp[i - 2] % mod + sum) % mod;
            }
            return (int) (dp[num_people] % mod);
        }
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
