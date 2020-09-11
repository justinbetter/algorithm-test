package com.justinbetter.algorithhm;

import java.util.HashMap;

public class StringProblem {

    public static void main(java.lang.String[] args) {
        // write your code here
        System.out.println(lengthOfLongestSubstring("aaabcssds"));
    }

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
}
