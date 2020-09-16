package com.justinbetter.algorithhm;

import java.util.*;

public class ArraySortProblem {

    public static void main(String[] args) {
        // write your code here
        int[] nums = new int[]{1, 32, 1, 2, 4, 5, 6, 0, 19};
        System.out.println(new findKthLargestSolution().findKthLargest2(nums, 2));
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

}
