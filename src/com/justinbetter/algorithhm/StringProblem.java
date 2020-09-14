package com.justinbetter.algorithhm;

import java.util.*;

public class StringProblem {

    public static void main(java.lang.String[] args) {
        // write your code here
        System.out.println(lengthOfLongestSubstring("aaabcssds"));
    }

    //复原IP地址
    class restoreIpAddressesSolution {
        static final int SEG_COUNT = 4;
        List<String> ans = new ArrayList<String>();
        int[] segments = new int[SEG_COUNT];

        public List<String> restoreIpAddresses(String s) {
            segments = new int[SEG_COUNT];
            dfs(s, 0, 0);
            return ans;
        }

        public void dfs(String s, int segId, int segStart) {
            //回溯结束条件
            // 如果找到了 4 段 IP 地址并且遍历完了字符串，那么就是一种答案
            if (segId == SEG_COUNT) {
                if (segStart == s.length()) {
                    StringBuffer ipAddr = new StringBuffer();
                    for (int i = 0; i < SEG_COUNT; ++i) {
                        ipAddr.append(segments[i]);
                        if (i != SEG_COUNT - 1) {
                            ipAddr.append('.');
                        }
                    }
                    ans.add(ipAddr.toString());
                }
                return;
            }

            // 如果还没有找到 4 段 IP 地址就已经遍历完了字符串，那么提前回溯
            if (segStart == s.length()) {
                return;
            }

            // 由于不能有前导零，如果当前数字为 0，那么这一段 IP 地址只能为 0
            if (s.charAt(segStart) == '0') {
                segments[segId] = 0;
                dfs(s, segId + 1, segStart + 1);
            }

            // 一般情况，枚举每一种可能性并递归
            int addr = 0;
            for (int segEnd = segStart; segEnd < s.length(); ++segEnd) {
                //获取该段的数字，判断符合255的规则，继续回溯
                addr = addr * 10 + (s.charAt(segEnd) - '0');
                if (addr > 0 && addr <= 0xFF) { //0xFF is an equal int(255).
                    segments[segId] = addr;
                    dfs(s, segId + 1, segEnd + 1);
                } else {
                    break;
                }
            }
        }
    }

    //简化路径
    class simplifyPathSolution {
        public String simplifyPath(String path) {
            Deque<String> stack = new LinkedList<>();
            for (String item : path.split("/")) {
                if (item.equals("..")) {
                    if (!stack.isEmpty()) stack.pop();
                } else if (!item.isEmpty() && !item.equals(".")) stack.push(item);
            }
            String res = "";
            for (String d : stack) res = "/" + d + res;
            return res.isEmpty() ? "/" : res;
        }
    }

    //翻转字符串里的单词
    class reverseWordsSolution {
        public String reverseWords(String s) {
            s = s.trim(); // 删除首尾空格
            int j = s.length() - 1, i = j;
            StringBuilder res = new StringBuilder();
            while (i >= 0) {
                while (i >= 0 && s.charAt(i) != ' ') i--; // 搜索首个空格
                res.append(s.substring(i + 1, j + 1) + " "); // 添加单词
                while (i >= 0 && s.charAt(i) == ' ') i--; // 跳过单词间空格
                j = i; // j 指向下个单词的尾字符
            }
            return res.toString().trim(); // 转化为字符串并返回
        }
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
