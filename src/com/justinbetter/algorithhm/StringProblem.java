package com.justinbetter.algorithhm;

import java.util.HashMap;

public class StringProblem {

    public static void main(java.lang.String[] args) {
        // write your code here
        System.out.println(lengthOfLongestSubstring("aaabcssds"));
    }

    class SolutionMultiply {

        public String multiply(String num1, String num2) {
            if (num1.equals("0") || num2.equals("0")) {
                return "0";
            }
            String ans = "0";
            int m = num1.length(), n = num2.length();
            //除数倒序相乘
            for (int i = n - 1; i >= 0; i--) {
                StringBuffer curr = new StringBuffer();
                int add = 0;
                //补0
                for (int j = n - 1; j > i; j--) {
                    curr.append(0);
                }
                //获取被除数
                int y = num2.charAt(i) - '0';
                //依次相乘
                for (int j = m - 1; j >= 0; j--) {
                    //获取被除数
                    int x = num1.charAt(j) - '0';
                    int product = x * y + add;
                    //添加余数
                    curr.append(product % 10);
                    //添加整数作为进位
                    add = product / 10;
                }
                if (add != 0) {
                    curr.append(add % 10);
                }
                ans = addStrings(ans, curr.reverse().toString());
            }
            return ans;
        }
        //字符串相加
        public String addStrings(String num1, String num2) {
            int i = num1.length() - 1, j = num2.length() - 1, add = 0;
            StringBuffer ans = new StringBuffer();
            while (i >= 0 || j >= 0 || add != 0) {
                int x = i >= 0 ? num1.charAt(i) - '0' : 0;
                int y = j >= 0 ? num2.charAt(j) - '0' : 0;
                int result = x + y + add;
                ans.append(result % 10);
                add = result / 10;
                i--;
                j--;
            }
            ans.reverse();
            return ans.toString();
        }
    }


    //获取最长不重复子序列
    static int lengthOfLongestSubstring(String s) {
        //思路：滑动窗口
        // i j 滑动判断字符重复,重复则跳到该边界之外，因为在这个字符内不可能出现更小的子序列了
        int length = s.length();
        int res = 0;
        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0, j = 0; j < length; j++) {
            if (map.containsKey(s.charAt(j))) {
                //move i 移到该重复的位置，因为在之前的字符内不可能出现更小的子序列了
                i = Math.max(map.get(s.charAt(j)), i);
            }
            res = Math.max(res, j - i + 1);
            map.put(s.charAt(j), j + 1);

        }
        return res;
    }

    //字符串的排列
    static class SolutionCheckinclusion {

        public boolean checkInclusion(String s1, String s2) {
            //保存每个字母出现的次数
            int length1 = s1.length();
            int length2 = s2.length();
            if (length1 > length2) {
                return false;
            }
            int[] c1 = new int[26];
            int[] c2 = new int[26];

            //填满排列1的位置
            for (int i = 0; i < length1; i++) {
                c1[s1.charAt(i) - 'a']++;
                c2[s2.charAt(i) - 'a']++;
            }

            for (int i = length1; i < length2; i++) {
                if (mathches(c1, c2)) {
                    return true;
                }
                c2[s2.charAt(i - length1) - 'a']--;
                c2[s2.charAt(i) - 'a']++;
            }
            return mathches(c1, c2);
        }

        //比较字母出现次数
        public boolean mathches(int[] a, int[] b) {
            for (int i = 0; i < 26; i++) {
                if (a[i] != b[i]) {
                    return false;
                }
            }
            return true;
        }
    }

    //最长公共前缀
    static class SolutionLongestCommonPrefix {
        public String longestCommonPrefix(String[] strs) {
            if (strs == null || strs.length == 0) {
                return "";
            } else {
                return longestCommonPrefix(strs, 0, strs.length - 1);
            }
        }

        public String longestCommonPrefix(String[] strs, int start, int end) {
            if (start == end) {
                return strs[start];
            } else {
                int mid = (end - start) / 2 + start;
                String lcpLeft = longestCommonPrefix(strs, start, mid);
                String lcpRight = longestCommonPrefix(strs, mid + 1, end);
                return commonPrefix(lcpLeft, lcpRight);
            }
        }

        public String commonPrefix(String lcpLeft, String lcpRight) {
            int minLength = Math.min(lcpLeft.length(), lcpRight.length());
            for (int i = 0; i < minLength; i++) {
                if (lcpLeft.charAt(i) != lcpRight.charAt(i)) {
                    return lcpLeft.substring(0, i);
                }
            }
            return lcpLeft.substring(0, minLength);
        }
    }

}
