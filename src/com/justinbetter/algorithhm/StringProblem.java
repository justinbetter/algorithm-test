package com.justinbetter.algorithhm;

import java.util.*;

public class StringProblem {

    public static void main(java.lang.String[] args) {
        // write your code here
        String s1 = "abcabcdssds";
        String s2 = "asdaabcdasd";

        System.out.println(new longestCommonSubsequenceSolution().longestCommonSubsequence2(s1, s2));
    }

    class findAnagramsSolution {
        public List<Integer> findAnagrams(String s, String p) {
            char[] arrS = s.toCharArray();
            char[] arrP = p.toCharArray();

            // 接收最后返回的结果
            List<Integer> ans = new ArrayList<>();

            // 定义一个 needs 数组来看 arrP 中包含元素的个数
            int[] needs = new int[26];
            // 定义一个 window 数组来看滑动窗口中是否有 arrP 中的元素，并记录出现的个数
            int[] window = new int[26];

            // 先将 arrP 中的元素保存到 needs 数组中
            for (int i = 0; i < arrP.length; i++) {
                needs[arrP[i] - 'a'] += 1;
            }

            // 定义滑动窗口的两端
            int left = 0;
            int right = 0;

            // 右窗口开始不断向右移动
            while (right < arrS.length) {
                int curR = arrS[right] - 'a';
                right++;
                // 将右窗口当前访问到的元素 curR 个数加 1
                window[curR] += 1;

                // 当 window 数组中 curR 比 needs 数组中对应元素的个数要多的时候就该移动左窗口指针
                while (window[curR] > needs[curR]) {
                    int curL = arrS[left] - 'a';
                    left++;
                    // 将左窗口当前访问到的元素 curL 个数减 1
                    window[curL] -= 1;
                }

                // 这里将所有符合要求的左窗口索引放入到了接收结果的 List 中
                if (right - left == arrP.length) {
                    ans.add(left);
                }
            }
            return ans;
        }
    }

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
                map.get(_str).add(strItem);
            }
        }
        return new ArrayList<>(map.values());

    }

    //LC22 括号生成
    class generateParenthesisSolution {
        public List<String> generateParenthesis(int n) {
            //DFS , res 判断剩余的左右括号是否满足
            List<String> res = new ArrayList<>();
            //执行深度优先遍历，搜索可能的结果
            dfs("",n,n,res);
            return res;
        }
        public void dfs(String temp,int left,int right,List<String> res){
            //在递归终止的时候，直接把它添加到结果集即可
            if (left == 0 && right == 0) {
                res.add(temp);
            }

            // 剪枝（左括号可以使用的个数严格大于右括号可以使用的个数，才剪枝，注意这个细节）
            if (left > right) {
                return;
            }
            if (left > 0) {
                dfs(temp+"(",left-1,right,res);
            }
            if (right > 0) {
                dfs(temp+")",left,right-1,res);
            }
        }

    }

    class isValidSolution {

        private final Map<Character,Character> map = new HashMap<Character,Character> (){
            {
                put('}','{');
                put(']','[');
                put(')','(');
            }
        };
        public boolean isValid(String s) {
            //辅助栈
            Stack<Character> stack = new Stack<Character>();

            for (int i =0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (this.map.containsKey(c)) {
                    char topElement  = stack.isEmpty()? '#' : stack.pop();
                    if (topElement != this.map.get(c)) {
                        return false;
                    }

                }else {
                    stack.push(c);
                }
            }
            return stack.isEmpty();

        }
    }

    class letterCombinationsSolution {
        List<String> res = new ArrayList<String>();
        Map<Character, String> phoneMap = new HashMap<Character, String>() {{
            put('2', "abc");
            put('3', "def");
            put('4', "ghi");
            put('5', "jkl");
            put('6', "mno");
            put('7', "pqrs");
            put('8', "tuv");
            put('9', "wxyz");
        }};
        public List<String> letterCombinations(String digits) {
            //回溯
            if (digits.length() == 0) {
                return res;
            }
            backTrack(digits,new StringBuffer(),0);
            return res;
        }
        public void backTrack(String digits,StringBuffer combination,int index) {
            if (index == digits.length()) {
                res.add(combination.toString());
            }else {
                Character _char = digits.charAt(index);
                String letters = phoneMap.get(_char);
                for (int i=0; i< letters.length();i++) {
                    combination.append(letters.charAt(i));
                    backTrack(digits,combination,index+1);
                    combination.deleteCharAt(index);
                }
            }
        }
    }

    class longestPalindromeSolution {
        public String longestPalindrome(String s) {
            //P(i,j)=P(i+1,j−1) S[i] = S[j]
            //动态规划：定义状态 子串是否为回文
            //状态转移: 头尾字符相等、边界情况考虑转移
            //考虑输出  d[i][j] =  true
            //考虑状态压缩
            int len = s.length();
            if (len < 2) {
                return s;
            }
            boolean[][] dp = new boolean[len][len];

            for (int i=0;i < len; i++) {
                dp[i][i] = true;
            }

            int  maxLen = 1;
            int start = 0;

            for (int j = 1; j < len; j++) {
                for (int i = 0; i < j; i++) {
                    if (s.charAt(i) == s.charAt(j)) {
                        if (j - i < 3) {
                            dp[i][j] = true;
                        } else{
                            dp[i][j] = dp[i+1][j -1];
                        }
                    } else {
                        dp[i][j] = false;
                    }

                    if (dp[i][j]) {
                        int curLen = j - i +1;
                        if (curLen > maxLen) {
                            maxLen = curLen;
                            start = i;
                        }
                    }
                }

            }
            return s.substring(start, start + maxLen);
        }
    }

    //LC438 找到子字符串中的异位词（双指针、滑动窗口）
    class Solution {
        public List<Integer> findAnagrams(String s, String p) {
            //mine：没想到
            //other：双指针构造滑动窗口，ans返回的条件有两个：
            //1. 当前字母的数组值和异位词数量相等；2. 同时窗口长度要和异位词相等
            char[] arrS = s.toCharArray();
            char[] arrP = p.toCharArray();

            // 接收最后返回的结果
            List<Integer> ans = new ArrayList<>();

            // 定义一个 needs 数组来看 arrP 中包含元素的个数
            int[] needs = new int[26];
            // 定义一个 window 数组来看滑动窗口中是否有 arrP 中的元素，并记录出现的个数
            int[] window = new int[26];

            // 先将 arrP 中的元素保存到 needs 数组中
            for (int i = 0; i < arrP.length; i++) {
                needs[arrP[i] - 'a'] += 1;
            }

            // 定义滑动窗口的两端
            int left = 0;
            int right = 0;

            // 右窗口开始不断向右移动
            while (right < arrS.length) {
                int curR = arrS[right] - 'a';
                right++;
                // 将右窗口当前访问到的元素 curR 个数加 1
                window[curR] += 1;

                // 当 window 数组中 curR 比 needs 数组中对应元素的个数要多的时候就该移动左窗口指针
                while (window[curR] > needs[curR]) {
                    int curL = arrS[left] - 'a';
                    left++;
                    // 将左窗口当前访问到的元素 curL 个数减 1
                    window[curL] -= 1;
                }

                // 这里将所有符合要求的左窗口索引放入到了接收结果的 List 中
                if (right - left == arrP.length) {
                    ans.add(left);
                }
            }
            return ans;
        }
    }


    //LC394 字符串解码
    class decodeStringSolution {
        public String decodeString(String s) {
            //mine: 根据要求解码，判断数字 Character.isDigital()，循环数字后的字母
            //问题是嵌套处理，如果再次遇到数字
            //other: 利用栈数据结构，关键是[入栈，]出栈
            LinkedList<Integer> stack_nums = new LinkedList<>();
            LinkedList<String> stack_strings = new LinkedList<>();

            StringBuilder res = new StringBuilder();
            int multi = 0;
            char[] chars = s.toCharArray();
            for (char c : chars) {
                if (c == '[') {
                    stack_nums.addLast(multi);
                    stack_strings.addLast(res.toString());
                    multi = 0;
                    res  =  new StringBuilder();
                }else if (c == ']') {
                    StringBuilder temp = new StringBuilder();
                    int num = stack_nums.removeLast();
                    for (int i=0; i < num;i++) {
                        temp.append(res.toString());
                    }
                    res = new StringBuilder(stack_strings.removeLast()+temp);

                }else if (c >= '0' && c <= '9') {
                    multi = multi*10 + Integer.parseInt(c+"");
                }else {
                    res.append(c);
                }
            }
            return res.toString();
        }
    }

    static class longestCommonSubsequenceSolution {
        //dp[i][j]的含义是，在必须把str1[i]和str2[j]当作公共子串最后一个字符的情况下，公共子串最长能有多长
        //遍历dp找到最大值及其位置，最长公共子串自然可以得到
        public String longestCommonSubsequence2(String text1, String text2) {
            char[] char1 = text1.toCharArray();
            char[] char2 = text2.toCharArray();
            int len1 = char1.length;
            int len2 = char2.length;
            int[][] dp = new int[len1][len2];
            int end = 0;
            int max = 0;
            //dp[0][0] = 0; int 默认是0
            for (int i = 0; i < len1; i++) {
                if (char1[i] == char2[0]) {
                    dp[i][0] = 1;
                }
            }
            for (int j = 0; j < len2; j++) {
                if (char2[j] == char1[0]) {
                    dp[0][j] = 1;
                }
            }
            for (int i = 1; i < len1; i++) {
                for (int j = 1; j < len2; j++) {
                    if (char1[i] == char2[j]) {
                        dp[i][j] = dp[i - 1][j - 1] + 1;
                        if (dp[i][j] > max) {
                            end = i;
                            max = dp[i][j];
                        }
                    }
                }
            }
//            return max;
            return text1.substring(end - max + 1, end + 1);
        }
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
