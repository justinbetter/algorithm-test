package com.justinbetter.algorithhm;

import java.text.DecimalFormat;

public class binarySearch {

    public static void main(String[] args) {
        // write your code here
        int number = 2;
//        System.out.println(mySqrt2(number, 0.0001));
//        System.out.println(SqrtByNewton(number, 0.001));
        int[] nums = {1, 3, 6, 10};
        System.out.println(new binarySearch().binary_searchHj1(nums, 10));
    }

    //√2 = 1.41421
    static Double mySqrt2(int x, double precision) {
        double left = 0;
        double right = (double) x;
        double lastMid = right;
        double mid = (left + right) / 2.0;
        // 判断精确
        while (abs(mid - lastMid) > precision) {
            if (mid * mid > x) {
                right = mid;
            } else {
                left = mid;
            }
            lastMid = mid;
            mid = (left + right) / 2.0;
        }
        return mid;
//        DecimalFormat df = new DecimalFormat("0.00");
//        return df.format(mid);
    }

    static double SqrtByNewton(int n, double precision) {
        double val = n; //最终
        double last; //保存上一个计算的值
        do {
            last = val;
            val = (val + n / val) / 2;
        } while (Math.abs(val - last) > precision);
        return val;
    }

    public static double abs(double a) {
        return (a <= 0.0D) ? 0.0D - a : a;
    }

    static int mySqrt(int x) {
        //二分法
        int left = 0;
        int right = x;
        int ans = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if ((long) mid * mid > x) {
                right = mid - 1;
            } else {
                ans = mid;
                left = mid + 1;
            }
        }
        return ans;

    }

    int binary_searchHj1(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                return mid;
            }
        }
        return -1;
    }

    int binary_search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] == target) {
                // 直接返回
                return mid;
            }
        }
        // 直接返回
        return -1;
    }

    int left_bound(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] == target) {
                // 别返回，锁定左侧边界
                right = mid - 1;
            }
        }
        // 最后要检查 left 越界的情况
        if (left >= nums.length || nums[left] != target)
            return -1;
        return left;
    }


    int right_bound(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] == target) {
                // 别返回，锁定右侧边界
                left = mid + 1;
            }
        }
        // 最后要检查 right 越界的情况
        if (right < 0 || nums[right] != target)
            return -1;
        return right;
    }

    int searchLeft(int[] nums, int target) {
        int left = 0;
        int right = nums.length -1;
        while(left <= right) {
            int mid = left + (right-left)/2;
            if (nums[mid] == target) {
                right = mid - 1;
            }else if (nums[mid] > target) {
                right = mid - 1;;
            }else {
                left = mid + 1;
            }
        }
        if (left < nums.length && nums[left] == target) {
            return left;
        }
        return -1;
    }

    int searchRight(int[] nums, int target) {
        int left = 0;
        int right = nums.length -1;
        while(left <= right) {
            int mid = left + (right-left)/2;
            if (nums[mid] == target) {
                left = mid+1;
            }else if (nums[mid] > target) {
                right = mid - 1;;
            }else {
                left = mid + 1;
            }
        }
        if (right >= 0 && nums[right] == target) {
            return right;
        }
        return -1;
    }
}
