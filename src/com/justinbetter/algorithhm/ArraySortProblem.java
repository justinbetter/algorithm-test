package com.justinbetter.algorithhm;

import java.util.*;

public class ArraySortProblem {

    public static void main(String[] args) {
        // write your code here
//        int[] nums = new int[]{5, 7, 7, 8, 8, 10};
        System.out.println(12%10);
//        int[] nums = new int[]{1,2,3,4,5,6};
//        rotateSolution.rotate(nums, 3);
    }

    class topKFrequentSolution {
        public int[] topKFrequent(int[] nums, int k) {
            //hashmap 优先队列 数字 次数
            HashMap<Integer,Integer> map = new HashMap<>();
            for(int i=0; i< nums.length;i++) {
                int key = nums[i];
                map.put(key,map.getOrDefault(key,0)+1);
            }
            //int[]{key,count}
            PriorityQueue<int[]> queue = new PriorityQueue<int[]>(new Comparator<int[]>(){
                public int compare(int[] m,int[] n) {
                    return m[1] - n[1];//次数比较，次数小的在前面
                }
            });
            //和小根堆顶比较
            for (int _key : map.keySet()){
                int count = map.get(_key);
                if(queue.size() == k){
                    if (queue.peek()[1] < count){
                        queue.poll();
                        queue.offer(new int[]{_key,count});
                    }
                }else {
                    queue.offer(new int[]{_key,count});
                }
            }
            int[] ans = new int[k];
            for (int i=0;i<k;i++) {
                ans[i] = queue.poll()[0];
            }
            return ans;
        }
    }

    class maxProductSolution {
        public int maxProduct(int[] nums) {
            int max = Integer.MIN_VALUE;
            //  考虑有负数，需要维护一个最小值和最大值交换
            int imax = 1;
            int imin = 1;
            for(int i = 0;i < nums.length;i++) {
                if (nums[i] < 0) {
                    int temp = imax;
                    imax = imin;
                    imin = temp;
                }
                imax = Math.max(nums[i]*imax,nums[i]);
                imin = Math.min(nums[i]*imin,nums[i]);

                //记录每次的最大值，因为中间可能改变了
                max = Math.max(max,imax);
            }
            return max;
        }
    }

    class rotateRectangularSolution {
        public void rotate(int[][] matrix) {
            //转置矩阵
            int n = matrix.length;
            for (int i=0;i < n;i++) {
                for (int j=i;j < n;j++) {
                    int temp = matrix[i][j];
                    matrix[i][j] = matrix[j][i];
                    matrix[j][i] = temp;
                }
            }
            //左右反转矩阵列
            for (int i=0;i <n;i++){
                for(int j=0;j < n/2;j++) {
                    int temp = matrix[i][j];
                    matrix[i][j] = matrix[i][n-j-1];
                    matrix[i][n-j-1] = temp;
                }
            }
        }
    }

    class nextPermutationSolution {
        public void nextPermutation(int[] nums) {
            //从后向前 寻找第一个升序元素对i，j，选择降序中大于i中最接近的值
            int i = nums.length -2;
            while(i >= 0 && nums[i+1] <= nums[i]){
                i--;
            }
            if(i>=0){ //cover 没有找到升序的情况
                int j = nums.length -1;
                while (j >= 0 && nums[j] <= nums[i]){
                    j--;
                }
                swap(nums,i,j);
            }
            reverse(nums,i+1);
        }
        public void reverse(int[] nums,int start) {
            int i = start,j=nums.length-1;
            while(i < j) {
                swap(nums,i,j);
                i++;
                j--;
            }
        }
        public void swap(int[] nums,int i,int j) {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
        }
    }

    public int maxArea(int[] height) {
        //双指针缩减范围，移动短的
        int left = 0;
        int right = height.length-1;
        int res = 0;
        while(left < right) {
            if (height[left] < height[right]){
                res = Math.max(res,(right-left)*height[left++]);
            }else {
                res = Math.max(res,(right-left)*height[right--]);
            }
        }
        return res;
    }

    //LC200 岛屿数量
    class numIslandsSolution {

        boolean[][] visited;
        int m;
        int n;
        int[][] direct = new int[][]{{-1,0},{0,-1},{1,0},{0,1}};
        public int numIslands(char[][] grid) {
            //回溯 dfs visited direction
            m = grid.length;
            n = grid[0].length;
            visited = new boolean[m][n];
            int res = 0;
            for(int i=0;i < m;i++) {
                for (int j=0;j < n;j++) {
                    if(grid[i][j] == '1' && !visited[i][j]) {
                        dfs(i,j,grid);
                        res++;
                    }
                }
            }
            return res;
        }
        private void dfs(int i, int j,char[][] grid) {
            visited[i][j] = true;
            //开始移动
            for(int k=0;k < 4;k++) {
                int x = i+direct[k][0];
                int y = j + direct[k][1];
                if (x >= 0 && x < m && y >=0 && y < n && grid[x][y] == '1'&& !visited[x][y]) {
                    dfs(x,y,grid);
                }
            }

        }
    }

    public int lengthOfLongestSubstring(String s) {
        HashMap<Character,Integer> map = new HashMap<>();
        int res = 0;
        int left = 0;
        for (int i=0; i< s.length();i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left,map.get(s.charAt(i)));
            }
            res = Math.max(res,i-left+1);
            map.put(s.charAt(i),i+1);
        }
        return res;
    }


    //LC739 每日温度（获取比当前值更大的数，返回用距离组成新数组）
    class dailyTemperaturesSolution {
        public int[] dailyTemperatures(int[] temperatures) {
            //mine：暴力法：双循环遍历数组，获取比当前值大的第一个数字
            //other：使用栈，获取当前栈
            //每次放入栈元素，如果当前数字大于栈顶元素的数字，
            //那么一定是第一个大于栈顶元素的数，直接求出下标差就是二者的距离

            LinkedList<Integer> stack = new LinkedList();
            int[] res = new int[temperatures.length];
            for (int i = 0 ; i < temperatures.length; i++) {
                while(!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                    int preIndex = stack.pop();
                    res[preIndex] = i - preIndex;
                }
                stack.push(i);
            }
            return res;

        }
    }

    //LC581 最短无序连续子数组
    class findUnsortedSubarraySolution {
        public int findUnsortedSubarray(int[] nums) {
            //mine：知道连续非有序子数组，排序，整个数组升序
            //记录非升序的子数组长度，
            //other：先排序，比较左右边界不同的数字
            //2. 从左到右维持最大值，寻找右边界end
            int[] cloneNums = nums.clone();
            Arrays.sort(cloneNums);
            //注意初始化边界
            int min = nums.length;
            int max = 0;
            for (int i = 0; i < nums.length; i++) {
                if (cloneNums[i] != nums[i]) {
                    //注意：min、max 是左右索引
                    min = Math.min(min,i);
                    max = Math.max(max,i);
                }
            }
            return max - min >= 0 ? max - min + 1 : 0;
        }
    }

    //LC560 和为K的子数组（前缀和+hash表计数）
    class subarraySumSolution {
        public int subarraySum(int[] nums, int k) {
            //mine：动态规划？dp[i][j] = dp[i-1][j] || dp[i-1][k - j]
            //other: 魔怔了，前缀和加hash表 判断k的次数
            Map<Integer, Integer> preSumFreq = new HashMap<>();
            // 对于下标为 0 的元素，前缀和为 0，个数为 1
            preSumFreq.put(0, 1);

            int preSum = 0;
            int count = 0;
            for (int num : nums) {
                preSum += num;

                //关键：找到和当前前缀和相差k的前缀和，计数
                // 先获得前缀和为 preSum - k 的个数，加到计数变量里
                //如果 map 中存在 key 为「当前前缀和 - k」，说明这个之前出现的前缀和，
                //满足「当前前缀和 - 该前缀和 == k」，它出现的次数，累加给 count。
                if (preSumFreq.containsKey(preSum - k)) {
                    count += preSumFreq.get(preSum - k);
                }

                // 然后维护 preSumFreq 的定义
                preSumFreq.put(preSum, preSumFreq.getOrDefault(preSum, 0) + 1);
            }
            return count;

        }
    }

    //LC287 寻找（时间换空间的）重复数
    class findDuplicateSolution {
        public int findDuplicate(int[] nums) {
            //mine：首先想到hash表存储，但是这里有限制，所以要用二分法
            //other：通过抽屉原理：10个苹果放9个抽屉，因为这里范围是[1,n]
            //所以设置mid，把mid前的数量加起来发现 count > mid 说明重复数大概率在left,mid中
            //因为是重复数，重复了才会超过mid

            //注意：left = 1！
            int left = 1;
            int right = nums.length -1;

            //二分法
            while (left < right) {
                // int mid = (left + right) >> 2;
                int mid = left + (right - left)/2;
                int count = 0;
                for (int num : nums) {
                    if (num <= mid) {
                        count++;
                    }
                }

                if (count > mid) {
                    right = mid;
                }else {
                    left = mid+1;
                }
            }
            return left;

        }
    }

    //LC253 会议室时间安排数量
    class minMeetingRoomsSolution {

        public int minMeetingRooms(int[][] intervals) {
            //mine：判断有没有冲突，但是代码怎么写？数组首先看看能不能排序
            //other：把开始时间和结束时间放在一起排序，开始时间+1，结束时间-1
            //或者1. 先按照开始时间升序排列，2. 使用优先队列，将结束时间放入堆中,和开始时间比较

            List<int[]> list = new ArrayList<>();
            for (int i=0;i< intervals.length;i++) {
                list.add(new int[]{intervals[i][0],1});
                list.add(new int[]{intervals[i][1],-1});
            }
            Collections.sort(list,(a,b)->{
                if(a[0] == b[0]) {
                    return a[1] - b[1];
                }
                return a[0]-b[0];
            });
            int res = 0;
            int cur = 0;
            for (int i=0; i < list.size(); i++) {
                int[] array = list.get(i);
                cur += array[1];
                res = Math.max(res,cur);
            }
            return res;

        }

        // 或者1. 先按照开始时间升序排列，2. 使用优先队列，将结束时间放入堆中,和开始时间比较
        public int minMeetingRooms3(int[][] intervals) {
            if(intervals.length == 0) return 0;
            Arrays.sort(intervals,Comparator.comparingInt(o->o[0]));
            PriorityQueue<Integer> minHeap = new PriorityQueue<>();
            int rooms = 0;
            for(int i=0; i<intervals.length; i++) {
                minHeap.offer(intervals[i][1]);
                if (intervals[i][0] < minHeap.peek()) {
                    rooms ++;
                } else {
                    minHeap.poll();
                }
            }
            return rooms;
        }

    }

    public int findKthLargest(int[] nums, int k) {
        //mine：使用优先队列，最小堆，将len-k poll出来，剩下就是ans
        int len = nums.length;
        PriorityQueue<Integer> queue = new PriorityQueue<>(len,(a,b)->(a-b));
        for (int i = 0 ; i < len; i++) {
            queue.offer(nums[i]);
        }

        for (int i=0; i < len - k; i++) {
            queue.poll();
        }

        return queue.peek();
    }

    //LC240 搜索升序的二维矩阵
    class searchMatrixSolution {
        public boolean searchMatrix(int[][] matrix, int target) {
            //mine: 二分法，行，列
            //other：利用该矩阵升序性质，从左下角为起点搜索
            //如果当前>目标值，行--；当前< 目标值，列++；
            int row  = matrix.length -1;
            int column = 0;

            while(row >=0 && column < matrix[0].length) {
                if (matrix[row][column] > target) {
                    row--;
                }else if (matrix[row][column] < target) {
                    column++;
                }else{
                    return true;
                }
            }
            return false;
        }
    }

    //LC207 课程表 有向无环图 拓扑排序；
    class canFinishSolution {
        public boolean canFinish(int numCourses, int[][] prerequisites) {
            //有向无环图，拓扑排序
            //入度表：指有向图中某点作为图中边的终点的次数之和。
            //每次都选取入度为 0 的点加入拓扑队列中，再删除与这一点连接的所有边。
            int[] indegrees = new int[numCourses];
            //邻接表：用于存储图
            List<List<Integer>> adjacency = new ArrayList<>();
            Queue<Integer> queue = new LinkedList<>();
            for (int i=0; i < numCourses;i++) {
                adjacency.add(new ArrayList<>());
            }

            //获取课程的入度和邻接表
            for(int[] cp : prerequisites) {
                indegrees[cp[0]]++;
                adjacency.get(cp[1]).add(cp[0]);
            }

            //获取入度为0的课程
            for(int i = 0; i < numCourses; i++)
                if(indegrees[i] == 0) queue.add(i);

            // BFS TopSort.
            while(!queue.isEmpty()) {
                int pre = queue.poll();
                numCourses--;
                for(int cur : adjacency.get(pre))
                    if(--indegrees[cur] == 0) queue.add(cur);
            }

            return numCourses == 0;
        }
    }


    public ListNode reverseList(ListNode head) {
        //迭代、递归，说白了都是引用指向
        ListNode prev = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = prev;
            prev = cur;
            cur = next;
        }
        //注意：这里返回新的pre节点，因为cur=null啦！
        return prev;
    }
    //合并两个有序数组
    class merge2Solution {
        public void merge(int[] nums1, int m, int[] nums2, int n) {
            //mine：双指针，比较大小，但是交换逻辑有点复杂；从后往前遍历比较，赋值
            //other： 1. 增加额外数组 2. 直接放进尾部，对数组排序 3. 最后的从后往前放，谁大谁交换到后面
            int i = m - 1;
            int j = n -1;
            int end = nums1.length -1;
            while (j >= 0) {
                if (i >= 0 && nums1[i] > nums2[j] ) {
                    nums1[end--] = nums1[i--];
                }else {
                    // swap(nums2[j--],nums1[end--]);
                    nums1[end--] = nums2[j--];
                }
            }
        }
    }

    //LC189
    static class rotateSolution {
        public static void rotate(int[] nums, int k) {
            int n = nums.length;
            int[] newArr = new int[n];
            for (int i = 0; i < n; ++i) {
                newArr[(i + k) % n] = nums[i];
            }
            System.arraycopy(newArr, 0, nums, 0, n);
        }
    }


    //LC 79 回溯dfs 设置visited 设置direction
    class Solution {
        public boolean exist(char[][] board, String word) {
            //回溯判断 dfs是否等于字符，不符合返回上一状态
            //设置标记
            int m = board.length;
            int n = board[0].length;
            boolean[][] visited = new boolean[m][n];
            //设置遍历方向
            int[][] direction = new int[][]{{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (dfs(i, j, 0, direction, board, visited, word)) {
                        return true;
                    }
                }
            }
            return false;
        }

        private boolean dfs(int i, int j, int start, int[][] direction, char[][] board, boolean[][] visited, String word) {
            if (start == word.length() - 1) {
                return board[i][j] == word.charAt(start);
            }
            if (board[i][j] == word.charAt(start)) {
                visited[i][j] = true;
                for (int k = 0; k < 4; k++) {
                    int x = i + direction[k][0];
                    int y = j + direction[k][1];
                    //判断是否在边界,是否访问过
                    if (x >= 0 && x < board.length && y >= 0 && y < board[0].length && !visited[x][y]) {
                        boolean flag = dfs(x, y, start + 1, direction, board, visited, word);
                        if (flag) {
                            return true;
                        }
                    }
                }
                visited[i][j] = false;
            }
            return false;
        }

    }

    //LC75 颜色分类 双指针解决
    class sortColorsSolution {
        public void sortColors(int[] nums) {
            //双指针扫描:0交换头 2交换尾
            int left = 0;
            int right = nums.length;
            int i = 0; //循环变量
            while (i < right) {
                int val = nums[i];
                if (val == 0) { //和left交换
                    swap(nums, left, i);
                    i++;
                    left++;
                } else if (val == 1) {
                    i++;
                } else if (val == 2) { //和right交换
                    right--;
                    swap(nums, right, i);
                }
            }
        }

        private void swap(int[] nums, int left, int right) {
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
        }
    }

    //LC76 字集
    static class subsetsSolution {
        public List<List<Integer>> subsets(int[] nums) {
            //思路：回溯 dfs
            int len = nums.length;
            List<List<Integer>> res = new ArrayList<>();
            List<Integer> path = new ArrayList<>();
            dfs(0, res, path, len, nums);
            return res;
        }

        private void dfs(int begin, List<List<Integer>> res, List<Integer> path, int len, int[] nums) {
            res.add(new ArrayList(path));
            for (int i = begin; i < len; i++) {
                path.add(nums[i]);
                dfs(i + 1, res, path, len, nums);
                path.remove(path.size() - 1);
            }
        }

        public List<List<Integer>> subsets2(int[] nums) {
            List<List<Integer>> res = new ArrayList<>();
            backtrack(0, nums, res, new ArrayList<Integer>());
            return res;

        }

        private void backtrack(int i, int[] nums, List<List<Integer>> res, ArrayList<Integer> tmp) {
            res.add(new ArrayList<>(tmp));
            for (int j = i; j < nums.length; j++) {
                tmp.add(nums[j]);
                backtrack(j + 1, nums, res, tmp);
                tmp.remove(tmp.size() - 1);
            }
        }
    }

    //LC64 最小路径和
    class minPathSumSolution {
        public int minPathSum(int[][] grid) {
            //dp[i][j] = min(dp[i-1][j],dp[i][j-1])+grid[i][j]
            int m = grid.length;
            int n = grid[0].length;
            int[][] dp = new int[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (i == 0 && j == 0) {
                        dp[i][j] = grid[i][j];
                        continue;
                    }
                    //边界上只有一条路径
                    if (i == 0) {
                        dp[i][j] = dp[i][j - 1] + grid[i][j];
                    } else if (j == 0) {
                        dp[i][j] = dp[i - 1][j] + grid[i][j];
                    } else {
                        dp[i][j] = Math.min(dp[i][j - 1], dp[i - 1][j]) + grid[i][j];
                    }
                }
            }
            return dp[m - 1][n - 1];
        }
    }

    //LC 62 不同路径
    class uniquePathsSolution {
        public int uniquePaths(int m, int n) {
            //dp[i][j] = dp[i-1][j] + dp[i][j-1]
            //dp[i][j] 表示当前有多少路径
            //边界只有1条路径
            int[][] dp = new int[m][n];
            for (int i = 0; i < m; i++) {
                dp[i][0] = 1;
            }
            for (int i = 0; i < n; i++) {
                dp[0][i] = 1;
            }
            for (int i = 1; i < m; i++) {
                for (int j = 1; j < n; j++) {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
            return dp[m - 1][n - 1];
        }
    }

    //    leetcode55 跳跃游戏
    static class canJumpSolution {
        public boolean canJump(int[] nums) {
            //遍历判断 res为最远的地方 如果遍历中 发现有res >= nums.length,说明可以到达；否则return false；
            int res = 0;
            for (int i = 0; i < nums.length; i++) {
                if (i <= res) { //能够到达的地方，不然没法到达
                    res = Math.max(res, i + nums[i]);
                    if (res >= (nums.length - 1)) {
                        return true;
                    }
                }

            }
            return false;
        }
    }

    static class groupAnagramsSolution {
        public List<List<String>> groupAnagrams(String[] strs) {
            //按照字母排序后的结果=key, 因此异位字母获取的key是相等的
            HashMap<String, List<String>> map = new HashMap<>();
            for (int i = 0; i < strs.length; i++) {
                String strItem = strs[i];
                char[] charArray = strItem.toCharArray();
                Arrays.sort(charArray);
                String _str = String.valueOf(charArray);
                if (map.containsKey(_str)) {
                    map.get(_str).add(strItem);
                } else {
                    map.put(_str, new ArrayList<>());
                }
            }
            return new ArrayList<>(map.values());

        }
    }

    static class combinationSumSolution {
        public List<List<Integer>> combinationSum(int[] candidates, int target) {
            //回溯+深度遍历
            List<List<Integer>> res = new ArrayList<>();
            if (candidates.length == 0) {
                return res;
            }
            LinkedList<Integer> path = new LinkedList<>();
            dfs(res, path, 0, candidates, target);
            return res;
        }

        private void dfs(List<List<Integer>> res, LinkedList<Integer> path, int index, int[] candidates, int target) {
            //结束条件
            if (target < 0) {
                return;
            }
            if (target == 0) {
                //这里需要new，不然深度遍历中撤销选择时候把这里也置空了
                res.add(new ArrayList<>(path));
                return;
            }

            for (int i = index; i < candidates.length; i++) {
                path.addLast(candidates[i]);
                dfs(res, path, i, candidates, target - candidates[i]);
                path.removeLast();
            }

        }
    }

    //34. 在排序数组中查找元素的第一个和最后一个位置
    static class searchRangeSolution {
        public int[] searchRange(int[] nums, int target) {
            //二分查找 left right mid
            if (nums.length == 0) {
                return new int[]{-1, -1};
            }
            int firstIndex = getPosition(nums, target);
            int lastIndex = getLastPosition(nums, target);
            return new int[]{firstIndex, lastIndex};

        }

        public int getPosition(int[] nums, int target) {
            int left = 0;
            int right = nums.length - 1;
            while (left < right) {
                int mid = left + ((right - left) >> 1);
                if (nums[mid] > target) {
                    right = mid - 1;
                } else if (nums[mid] == target) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            if (nums[left] == target) {
                return left;
            }
            //没找到
            return -1;
        }

        public int getLastPosition(int[] nums, int target) {
            int left = 0;
            int right = nums.length - 1;
            while (left < right) {
                int mid = left + ((right - left + 1) >> 1);
                if (nums[mid] > target) {
                    right = mid - 1;
                } else if (nums[mid] == target) {
                    left = mid;
                } else {
                    left = mid + 1;
                }
            }
            if (left < nums.length && nums[left] == target) {
                return left;
            }

            return -1;
        }
    }

    //螺旋矩阵
    class spiralOrderSolution {
        //重点是变更方向
        // 设定上下左右边界，边界交错就截止
        public List<Integer> spiralOrder(int[][] matrix) {
            List<Integer> res = new LinkedList<>();
            if (matrix.length == 0) {
                return res;
            }
            //初始上下左右
            int up = 0, down = matrix.length - 1, left = 0, right = matrix[0].length - 1;
            while (true) {
                //左右遍历
                for (int col = left; col <= right; ++col) {
                    res.add(matrix[up][col]);
                }
                //上边界+1 如果超过down，边界交错，遍历结束
                if (++up > down) {
                    break;
                }
                for (int row = up; row <= down; ++row) {
                    res.add(matrix[row][right]);
                }
                if (--right < left) {
                    break;
                }
                for (int col = right; col >= left; --col) {
                    res.add(matrix[down][col]);
                }
                if (--down < up) {
                    break;
                }
                for (int row = down; row >= up; --row) {
                    res.add(matrix[row][left]);
                }
                if (++left > right) {
                    break;
                }
            }
            return res;
        }
    }


    //矩阵中的最长递增路径
    class longestIncreasingPathSolution {
        //深度优先遍历+备忘录
        //获取行列，深度遍历,记录ans 是最大的row column
        public int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        public int rows, columns;

        public int longestIncreasingPath(int[][] matrix) {
            if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
                return 0;
            }
            rows = matrix.length;
            columns = matrix[0].length;
            int[][] memo = new int[rows][columns];
            int ans = 0;
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < columns; ++j) {
                    ans = Math.max(ans, dfs(matrix, i, j, memo));
                }
            }
            return ans;
        }

        public int dfs(int[][] matrix, int row, int column, int[][] memo) {
            //备忘录
            //行列开始移动，设置newRow newColumn
            //记录最大递增序列： 0<新行<行数 0<新列<列数 新值 > column
            //表示记录过了
            if (memo[row][column] != 0) {
                return memo[row][column];
            }
            //记录下该位置
            ++memo[row][column];
            //开始移动
            for (int[] dir : dirs) {
                //行列移动
                int newRow = row + dir[0], newColumn = column + dir[1];
                //记录最大递增序列
                if (newRow >= 0 && newRow < rows && newColumn >= 0 && newColumn < columns && matrix[newRow][newColumn] > matrix[row][column]) {
                    memo[row][column] = Math.max(memo[row][column], dfs(matrix, newRow, newColumn, memo) + 1);
                }
            }
            return memo[row][column];
        }
    }

    // 接雨水
    //最两端的列不用考虑，因为一定不会有水。所以下标从 1 到 length - 2
//    注意雨水单位计算方式是该列左右两边最高的墙中 选择一个较矮的高度-当前列高度就是该列积累的雨水
//    该列大于等于较矮高度，不积水
//    该列小于较矮高度，积水=较矮高度-当前列高度
    class trapSolution {

        public int trap1(int[] height) {
            int sum = 0;
            //最两端的列不用考虑，因为一定不会有水。所以下标从 1 到 length - 2
            for (int i = 1; i < height.length - 1; i++) {
                int max_left = 0;
                //找出左边最高
                for (int j = i - 1; j >= 0; j--) {
                    if (height[j] > max_left) {
                        max_left = height[j];
                    }
                }
                int max_right = 0;
                //找出右边最高
                for (int j = i + 1; j < height.length; j++) {
                    if (height[j] > max_right) {
                        max_right = height[j];
                    }
                }
                //找出两端较小的
                int min = Math.min(max_left, max_right);
                //只有较小的一段大于当前列的高度才会有水，其他情况不会有水
                if (min > height[i]) {
                    sum = sum + (min - height[i]);
                }
            }
            return sum;
        }

        //动态规划法：左边最高的就是上一个最高的和当前左边第一个比较；右边同理；
        //循环列，计算雨水总和
        public int trap2(int[] height) {
            int sum = 0;
            int[] max_left = new int[height.length];
            int[] max_right = new int[height.length];

            for (int i = 1; i < height.length - 1; i++) {
                max_left[i] = Math.max(max_left[i - 1], height[i - 1]);
            }
            for (int i = height.length - 2; i >= 0; i--) {
                max_right[i] = Math.max(max_right[i + 1], height[i + 1]);
            }
            for (int i = 1; i < height.length - 1; i++) {
                int min = Math.min(max_left[i], max_right[i]);
                if (min > height[i]) {
                    sum = sum + (min - height[i]);
                }
            }
            return sum;
        }
    }

    //合并区间
    class mergeSolution {
        public int[][] merge(int[][] intervals) {
            List<int[]> res = new ArrayList<>();
            if (intervals.length == 0 || intervals == null) return res.toArray(new int[0][]);
            // 对起点终点进行排序
            Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
            int i = 0;
            //迭代区间元素
            while (i < intervals.length) {
                //获取区间左右端点
                int left = intervals[i][0];
                int right = intervals[i][1];
                // 如果有重叠，循环判断哪个起点满足条件
                //如果下一个区间的左端点小于当前区间的右端点，说明有重合，合并区间，设置有右端点为下一个区间的右端点；
                //不断循环下一个区间，有重合的就和当前区间合并
                while (i < intervals.length - 1 && intervals[i + 1][0] <= right) {
                    i++;
                    right = Math.max(right, intervals[i][1]);
                }
                // 将现在的区间放进res里面
                res.add(new int[]{left, right});
                // 接着判断下一个区间
                i++;
            }
            return res.toArray(new int[0][]);
        }
    }

    //朋友圈 DFS
    public class findCircleNumSolution {

        //广度遍历
        public int findCircleNum2(int[][] M) {
            int[] visited = new int[M.length];
            int count = 0;
            //队列
            Queue<Integer> queue = new LinkedList<>();
            //横向入队
            for (int i = 0; i < M.length; i++) {
                if (visited[i] == 0) {
                    //如果没有访问过加入队列
                    queue.add(i);
                    //开始广度搜索，针对i遍历行列
                    while (!queue.isEmpty()) {
                        //出队，获取刚才访问的节点
                        int s = queue.remove();
                        visited[s] = 1;
                        //纵向遍历，判断如果没有访问过，入队，进行横向遍历
                        for (int j = 0; j < M.length; j++) {
                            if (M[s][j] == 1 && visited[j] == 0)
                                queue.add(j);
                        }
                    }
                    count++;
                }
            }
            return count;
        }


        //一次遍历i所在的行和列
        public void dfs(int[][] M, int[] visited, int i) {
            //纵向遍历i，横向遍历j，如果该数字为1且没访问过，继续深度遍历，并将该数字位置记录visited
            for (int j = 0; j < M.length; j++) {
                if (M[i][j] == 1 && visited[j] == 0) {
                    visited[j] = 1;
                    //从j位置继续纵向遍历，因为是对称的，所以该行位置和该列值是相同的，等于遍历列了
                    dfs(M, visited, j);
                }
            }
        }

        //深度遍历
        public int findCircleNum(int[][] M) {
            //设置访问过的数组
            int[] visited = new int[M.length];
            //记录深度遍历次数
            int count = 0;
            //纵向遍历
            for (int i = 0; i < M.length; i++) {
                //第i行遍历，如果没有访问过，开始深度遍历该行
                if (visited[i] == 0) {
                    dfs(M, visited, i);
                    count++;
                }
            }
            return count;
        }
    }

    //第k个排列
    //深度遍历+剪枝
    class getPermutationSolution {

        //获取排列数 确定K范围
        public String getPermutation2(int n, int k) {
            this.n = n;
            this.k = k;
            //计算阶乘,获取所有的排列数
            int[] permutations = new int[n + 1];
            permutations[0] = 1;
            for (int i = 1; i <= n; i++) {
                permutations[i] = permutations[i - 1] * i;
            }
            used = new boolean[n + 1];
            Arrays.fill(used, false);
            //确定范围
            StringBuilder path = new StringBuilder();
            dfs2(0, path, permutations);
            return path.toString();
        }

        public void dfs2(int index, StringBuilder path, int[] permutations) {
            if (index == n) {
                return;
            }

            //剩下的排列数
            for (int i = 1; i <= n; i++) {
                int counts = permutations[n - 1 - index];
                if (used[i]) {
                    continue;
                }
                if (counts < k) { //不再排列中
                    k -= counts;
                    continue;
                }
                //在排列中
                path.append(i);
                used[i] = true;
                dfs2(index + 1, path, permutations);
                return;
            }

        }


        /**
         * 记录数字是否使用过
         */
        private boolean[] used;

        /**
         * 阶乘数组
         */
        private int[] factorial;

        private int n;
        private int k;

        public String getPermutation(int n, int k) {
            this.n = n;
            this.k = k;
            //计算阶乘，获取每个排列的个数，可以通过个数判断结果会在哪个范围
            calculateFactorial(n);

            // 查找全排列需要的布尔数组
            used = new boolean[n + 1];
            Arrays.fill(used, false);

            StringBuilder path = new StringBuilder();
            //遍历第一层
            dfs(0, path);
            return path.toString();
        }


        /**
         * @param index 在这一步之前已经选择了几个数字，其值恰好等于这一步需要确定的下标位置
         * @param path
         */
        private void dfs(int index, StringBuilder path) {
            //说明到最后的数字了，也就是遍历到叶子节点了，没有必要再继续了
            if (index == n) {
                return;
            }

            //计算还未确定的数字的全排列的个数，第 1 次进入的时候是 n - 1
            //获取排列的个数，由于是递归，index表示确定的排列数字，这里要排除，以便获取剩下的排列个数
            int cnt = factorial[n - 1 - index];
            //从 1到n 开始迭代
            for (int i = 1; i <= n; i++) {
                if (used[i]) {
                    continue;
                }
                //如果排列的个数小于k，说明k不在这个i的排列中，进入下次循环；
                //将k减去此次的排列数，便于与下次的排列数比较
                if (cnt < k) {
                    k -= cnt;
                    continue;
                }
                // 走到这里，说明这里的排列个数 >= k,加入排列的第一个数字，其他的数字继续递归；
                // 同时由于第一个数字已经用了，所以标记下，防止之后对这个数字重复计算
                path.append(i);
                used[i] = true;
                //进去子排列的递归
                dfs(index + 1, path);
                // 注意 1：不可以回溯（重置变量），算法设计是「一下子来到叶子结点」，没有回头的过程
                // 注意 2：这里要加 return，后面的数没有必要遍历去尝试了
                return;
            }
        }

        /**
         * 计算阶乘数组
         *
         * @param n
         */
        private void calculateFactorial(int n) {
            factorial = new int[n + 1];
            factorial[0] = 1;
            for (int i = 1; i <= n; i++) {
                factorial[i] = factorial[i - 1] * i;
            }
        }
    }

    //最长连续序列
    class longestConsecutiveSolution {
        public int longestConsecutive2(int[] nums) {
            HashSet<Integer> hs = new HashSet<>();
            for (int num : nums) {
                hs.add(num);
            }
            int longResult = 0;
            for (int i = 0; i < nums.length; i++) {
                if (!hs.contains(nums[i] - 1)) {
                    int currentNum = nums[i];
                    int currentLong = 1;
                    while (hs.contains(currentNum + 1)) {
                        currentLong++;
                        currentNum++;
                    }
                    longResult = Math.max(currentLong, longResult);
                }
            }
            return longResult;
        }

        //全部存入hash集合，迭代判断当前和之后+1的数是否在hash中，在的话+1，不在的话下次循环；
        //如果之前-1的hash在，说明前面的迭代计数过了，跳过，Math.max比较每次序列长度，迭代结束；返回最终的序列长度
        public int longestConsecutive(int[] nums) {
            Set<Integer> num_set = new HashSet<Integer>();
            for (int num : nums) {
                num_set.add(num);
            }

            int longestStreak = 0;

            for (int num : num_set) {
                if (!num_set.contains(num - 1)) {
                    int currentNum = num;
                    int currentStreak = 1;

                    while (num_set.contains(currentNum + 1)) {
                        currentNum += 1;
                        currentStreak += 1;
                    }

                    longestStreak = Math.max(longestStreak, currentStreak);
                }
            }

            return longestStreak;
        }
    }

    //数组中的第K个最大元素：堆排序
    //插入元素的过程，我们知道每次与n/(2^x)的位置进行比较，所以，插入元素的时间复杂度为O(log n)。插入元素时进行的堆化，也叫自下而上的堆化
    //删除了堆顶元素 我们可以把最后一个元素移到根节点的位置，开始自上而下的堆化
    static class findKthLargestSolution {
        public int findKthLargest2(int[] nums, int k) {
            //数组建max堆
            //删除k-1个堆顶元素
            int heapSize = nums.length;
            buildMaxHeap2(nums, heapSize);
            for (int i = nums.length - 1; i >= nums.length - k + 1; i--) {
                swap2(nums, i, 0);
                heapSize--;
                heapify2(nums, 0, heapSize);
            }
            return nums[0];
        }

        public void buildMaxHeap2(int[] nums, int heapSize) {
            //从数组的第一个非叶子节点开始
            //不停--，自上而下的堆化，最后建立最大堆
            for (int i = heapSize / 2; i >= 0; i--) {
                heapify2(nums, i, heapSize);
            }
        }

        public void heapify2(int[] nums, int i, int heapSize) {
            //获取左右节点，比较交换，继续递归
            int l = 2 * i + 1;
            int r = 2 * i + 2;
            int largest = i;
            if (l < heapSize && nums[l] > nums[largest]) {
                largest = l;
            }
            if (r < heapSize && nums[r] > nums[largest]) {
                largest = r;
            }
            if (largest != i) {
                swap2(nums, largest, i);
                heapify2(nums, largest, heapSize);
            }
        }

        public void swap2(int[] nums, int i, int j) {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
        }

        public int findKthLargest(int[] nums, int k) {
            int heapSize = nums.length;
            //建堆
            buildMaxHeap(nums, heapSize);
            //将最大数移到最后，等同于删除，剩下的元素继续堆化，获取第2个最大数，继续删除，到第k-1个，堆顶就是所求的第k个最大数
            for (int i = nums.length - 1; i >= nums.length - k + 1; --i) {
                swap(nums, 0, i);
                --heapSize;
                maxHeapify(nums, 0, heapSize);
            }
            return nums[0];
        }

        //建堆
        public void buildMaxHeap(int[] a, int heapSize) {
            //从数组的第一个非叶子节点开始
            //不停--，自上而下的堆化，最后建立最大堆
            for (int i = heapSize / 2; i >= 0; --i) {
                maxHeapify(a, i, heapSize);
            }
        }

        public void maxHeapify(int[] a, int i, int heapSize) {
            //获取左右子节点和当前节点比较
            int l = i * 2 + 1, r = i * 2 + 2, largest = i;
            //左子节点大于当前节点，准备交换
            if (l < heapSize && a[l] > a[largest]) {
                largest = l;
            }
            //右子节点大于当前节点，准备交换
            if (r < heapSize && a[r] > a[largest]) {
                largest = r;
            }
            //只要不是当前节点，开始交换
            if (largest != i) {
                swap(a, i, largest);
                //继续判断子节点的堆化，超过数组就停止递归了
                maxHeapify(a, largest, heapSize);
            }
        }

        public void swap(int[] a, int i, int j) {
            int temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
    }

    //滑动窗口：最长连续递增序列
    class findLengthOfLCISSolution {
        public int findLengthOfLCIS(int[] nums) {
            int ans = 0, anchor = 0;
            for (int i = 0; i < nums.length; ++i) {
                if (i > 0 && nums[i - 1] >= nums[i]) anchor = i;
                ans = Math.max(ans, i - anchor + 1);
            }
            return ans;
        }
    }

    //搜索旋转数组
    class searchSolution {
        public int search(int[] nums, int target) {
            if (nums == null || nums.length == 0) {
                return -1;
            }
            int start = 0;
            int end = nums.length - 1;
            int mid;
            while (start <= end) {
                mid = start + (end - start) / 2;
                if (nums[mid] == target) {
                    return mid;
                }
                //前半部分有序,注意此处用小于等于
                if (nums[start] <= nums[mid]) {
                    //target在前半部分
                    if (target >= nums[start] && target < nums[mid]) {
                        end = mid - 1;
                    } else {
                        start = mid + 1;
                    }
                } else {
                    if (target <= nums[end] && target > nums[mid]) {
                        start = mid + 1;
                    } else {
                        end = mid - 1;
                    }
                }

            }
            return -1;

        }
    }

    //岛屿面积
    class maxAreaOfIslandSolution {
        public int maxAreaOfIsland(int[][] grid) {
            int res = 0;
            for (int i = 0; i < grid.length; i++) {
                for (int j = 0; j < grid[i].length; j++) {
                    if (grid[i][j] == 1) {
                        res = Math.max(res, dfs(i, j, grid));
                    }
                }
            }
            return res;
        }

        // 每次调用的时候默认num为1，进入后判断如果不是岛屿，则直接返回0，就可以避免预防错误的情况。
        // 每次找到岛屿，则直接把找到的岛屿改成0，这是传说中的沉岛思想，就是遇到岛屿就把他和周围的全部沉默。
        // ps：如果能用沉岛思想，那么自然可以用朋友圈思想。有兴趣的朋友可以去尝试。
        private int dfs(int i, int j, int[][] grid) {
            if (i < 0 || j < 0 || i >= grid.length || j >= grid[i].length || grid[i][j] == 0) {
                return 0;
            }
            grid[i][j] = 0;
            int num = 1;
            num += dfs(i + 1, j, grid);
            num += dfs(i - 1, j, grid);
            num += dfs(i, j + 1, grid);
            num += dfs(i, j - 1, grid);
            return num;

        }
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
                if (i > 0 && nums[i] == nums[i - 1]) continue;
                int L = i + 1;
                int R = len - 1;
                while (L < R) {
                    int sum = nums[i] + nums[L] + nums[R];
                    if (sum == 0) {
                        ans.add(Arrays.asList(nums[i], nums[L], nums[R]));
                        while (L < R && nums[L] == nums[L + 1]) L++;
                        while (L < R && nums[R] == nums[R - 1]) R--;
                        L++;
                        R--;
                    } else if (sum < 0) {
                        L++;
                    } else if (sum > 0) {
                        R--;
                    }
                }
            }
            return ans;

        }
    }

    public class sortArraySolution {

        // 快速排序 1：基本快速排序

        /**
         * 列表大小等于或小于该大小，将优先于 quickSort 使用插入排序
         */
        private final int INSERTION_SORT_THRESHOLD = 7;

        private final Random RANDOM = new Random();


        public int[] sortArray(int[] nums) {
            int len = nums.length;
            quickSort(nums, 0, len - 1);
            return nums;
        }

        private void quickSort(int[] nums, int left, int right) {
            // 小区间使用插入排序
            if (right - left <= INSERTION_SORT_THRESHOLD) {
                insertionSort(nums, left, right);
                return;
            }
            //获取基准index，继续递归
            int pIndex = partition(nums, left, right);
            quickSort(nums, left, pIndex - 1);
            quickSort(nums, pIndex + 1, right);
        }

        /**
         * 对数组 nums 的子区间 [left, right] 使用插入排序
         *
         * @param nums  给定数组
         * @param left  左边界，能取到
         * @param right 右边界，能取到
         */
        private void insertionSort(int[] nums, int left, int right) {
            for (int i = left + 1; i <= right; i++) {
                //记录要插入的数据
                int temp = nums[i];
                int j = i;
                while (j > left && nums[j - 1] > temp) {
                    //当前值等于前一个值
                    nums[j] = nums[j - 1];
                    //设置当前值index为前一个
                    j--;
                }
                //当前值等于temp，如果有插入，已经将需要插入的数组更新到后一个值了
                nums[j] = temp;
            }
        }

        private int partition(int[] nums, int left, int right) {
            //快速排序
            int randomIndex = RANDOM.nextInt(right - left + 1) + left;
            swap(nums, left, randomIndex);

            // 基准值
            int pivot = nums[left];
            int lt = left;
            // 循环不变量：
            // all in [left + 1, lt] < pivot
            // all in [lt + 1, i) >= pivot
            //循环数组
            for (int i = left + 1; i <= right; i++) {
                //小于基准，放左边
                if (nums[i] < pivot) {
                    //继续搜索
                    lt++;
                    //交换和左指针的位置，保证左边都是小于值
                    swap(nums, i, lt);
                }
            }
            //遍历结束后，交换左指针和基准的位置，返回左指针
            swap(nums, left, lt);
            return lt;
        }

        private void swap(int[] nums, int index1, int index2) {
            int temp = nums[index1];
            nums[index1] = nums[index2];
            nums[index2] = temp;
        }
    }

}
