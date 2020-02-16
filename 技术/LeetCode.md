# LeetCode

+ 题解网站[leetcode](https://github.com/azl397985856/leetcode)
+ 可视化网站[LeetCodeAnimation](https://link.zhihu.com/?target=https%3A//github.com/MisterBooo/LeetCodeAnimation)
+ 面试难度在LeetCode的中等难度水平，应达到20分钟一题

## [Top 100 Liked Questions](https://leetcode-cn.com/problemset/hot-100/)

### [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

+ 题目：在list中找到两个数，满足其和为target。难点为降低时间复杂度。

+ 思路

  + 暴力法较慢，$O(n^2)$时间复杂度。
  + **哈希（字典）**，边存边遍历，注意`twoSum([3,3], 6)`测试点，$O(n)$时间复杂度。
  
+ 题解

  ```python
  class Solution:
      def twoSum(self, nums: List[int], target: int) -> List[int]:
          nums_index = {}  # 存储num第一次出现的index，防止twoSum([3,3], 6)的测试点
          for i, num in enumerate(nums):  # 边存边遍历
              j = nums_index.get(target - num)
              if j != None:
                  return [j, i]
              nums_index[num] = i
  ```

+ 知识点
  + 拷贝
    + 赋值：`list2 = list1`，即别名，变一个list的元素时另一list也会**改变**。
    + 浅拷贝：`list2 = list1.copy()`，变一个list的元素时另一list**不变**，变一个list的元素中的元素（子元素）时另一list**改变**。
    + 深拷贝：`list2 = copy.deepcopy(list1)`，变一个list的元素时另一list**不变**，完全独立。
  + 同时遍历list的index和value：`for index, value in enumerate(list)`。
  + python字典用哈希实现，对于哈希，一般复杂度为$O(1)$，最差情况全部哈希为一样的值，复杂度为$O(n)$。

### [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

+ 题目：将链表各项相加，注意进位。难点为考虑进位。

+ 思路：逐位相加，注意`(2) + (8)`测试点。

+ 题解

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
          carried = 0  # 进位
          head = ListNode(0)
          last = head
          while l1 or l2:
              temp = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carried 
              last.next = ListNode(temp % 10)
              last = last.next
              carried = temp // 10
              if l1:
                  l1 = l1.next
              if l2:
                  l2 = l2.next
          if carried:
              last.next = ListNode(carried)
          return head.next
  ```

+ 知识点
  + python中简写`if a != None`或`if a != 0`为`if a`。
  + python中没有c语言的三目表达式，应写成`a = b if b > c else c`。

### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

+ 题目：找到一个字符串内的最长字串，满足字串内没有重复的字母。难点为降低时间复杂度。

+ 思路

  + 暴力法，找到以每个`string[index]`开头的满足条件子串，$O(n^2)$时间复杂度。
  + ==**滑动窗口**==，即在字符串中维护一个满足条件的窗口`string[start, end]`，$O(n)$时间复杂度。
  
+ 题解

  ```python
  class Solution:
      def lengthOfLongestSubstring(self, s: str) -> int:
          res = ''  # 滑动窗口
          max_len = 0
          for char in s:
              if char not in res:
                  res += char
                  if len(res) > max_len:
                      max_len = len(res)
              else:
                  res = res[res.index(char) + 1:]
                  res += char
          return max_len
  ```

### [4. 寻找两个有序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

+ 题目：给两个各自有序的list，找到合并后list的中位数。难点为要求$O(\log(m+n))$时间复杂度。

+ 思路

  + 合并两个list，每次从首位各pop一个元素。$O(m+n)$时间复杂度，不符合。

  + ==**二分查找**==：看到复杂度为$\log$，显然需要二分思想。题目变为在$O(\log(m+n))$时间复杂度内找第k小的数。
  + 首先保证`nums1`短于`nums2`，此时找第`k`小的数，只需判断`nums1[k/2]`和`nums2[k/2]`谁更小，若前者更小，则`nums1[:k/2+1]`的`x`个数一定小于总数组第`k`小个数，将这一部分移除，只需找到剩余数组的第`k-x`小的数。直到需要找到剩余部分第一小的数，比较两数组首元素即可。

+ 题解

  ```python
  def minKth(nums1: List[int], start1: int, end1: int, nums2: List[int], start2: int, end2: int, k: int) -> int:  # 求第k小的数
      while 1:
          if end1 - start1 > end2 - start2:  # 保证nums1最短
              return minKth(nums2, start2, end2, nums1, start1, end1 ,k)
          if end1 - start1 < 0:  # 在nums1最短前提下，nums2不可能先于nums1空掉，则只有可能nums1为空
              return nums2[start2 + k - 1]
          if k == 1:
              return min(nums1[start1], nums2[start2])  # 要找第1小的数时，只需要比较nums1和nums2第一个数
          i = start1 + min(end1 - start1 + 1, k // 2) - 1
          j = start2 + min(end2 - start2 + 1, k // 2) - 1
          if nums1[i] <= nums2[j]:  # start1到i的数一定小于总数组第k小的数，移走x个数，此后只需要找剩余数组里第k-x小的数
              k -= i - start1 + 1
              start1 = i + 1
          else:
              k -= j - start2 + 1
              start2 = j + 1
  
  class Solution:
      def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
          if len(nums1) > len(nums2):  # 保证nums1最短
              nums1, nums2 = nums2, nums1
          m, n = len(nums1), len(nums2)
          left = (m + n + 1) // 2  # 对于m+n为奇数，第left小和第right小的数是一个数
          right = (m + n + 2) // 2
          return (minKth(nums1, 0, m - 1, nums2, 0, n - 1, left) + minKth(nums1, 0, m - 1, nums2, 0, n - 1, right)) / 2
  ```


### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

+ 题目：找到一个string内的最长回文子串。

+ 思路

  + ==**动态规划**==，状态转移方程$dp\{i,j\}=\begin{cases}true&&dp\{i+1,j-1\}\ is\ true\ and\ s[i]==s[j]\\false&& other\end{cases}$，即`dp{i,j}=(s[i]==s[j]) and dp{i+1,j-1}`，$O(n^2)$时间复杂度。显然`j>=i`，故`dp`矩阵为上三角矩阵，且`dp{i+1,j-1}`在`dp{i,j}`左下方向，故遍历时从右下往左上遍历，即外`i--`内`j++`。
  + 中心扩展法，根据回文中心遍历，注意回文中心可能为`abcba`的`c`，也可能为`abba`的`bb`，$O(n^2)$时间复杂度。

+ 题解

  ```python
  class Solution:  # 动态规划
      def longestPalindrome(self, s: str) -> str:
          length = len(s)
          if length < 2:
              return s
  
          res = ""
          dp = [[False] * length for i in range(length)]
          for i in range(length):
              if i < length - 1 and s[i] == s[i + 1]:
                  dp[i][i + 1] = True
              dp[i][i] = True
          for i in range(length - 1, -1, -1):
              for j in range(i, length):
                  if i < length - 1 and j > 0 and i + 1 <= j - 1:
                      dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1]
                  if dp[i][j] and j - i + 1 > len(res):
                      res = s[i : j + 1]
          return res
  
  
  class Solution:  # 中心扩展法
      def longestPalindrome(self, s: str) -> str:
          length = len(s)
          if length < 2:
              return s
  		
          res = ""
          for index in range(length):
              for bias in range(min(index + 1, length - index)):
                  if s[index - bias] == s[index + bias]:
                      if bias + bias + 1 > len(res):
                          res = s[index - bias : index + bias + 1]
                  else:
                      break
              if index != length - 1:
                  for bias in range(min(index + 1, length - index - 1)):
                      if s[index - bias] == s[index + 1 + bias]:
                          if bias + bias + 2 > len(res):
                              res = s[index - bias : index + bias + 2]
                      else:
                          break
          return res
  ```

+ 知识点

  + 初始化二维数组`[[0] * 5 for i in range(5)]`或`[[0 for i in range(5)] for i in range(5)]`。

### [10. 正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)



## [Top Interview Questions](https://leetcode-cn.com/problemset/top/)

### 拓扑排序：[296.火星词典](https://leetcode-cn.com/problems/alien-dictionary/comments/)

+ 对于一个图，满足每个节点只出现一次，A节点在B节点前则只有A到B的边没有B到A的边，则符合拓扑排序。输出时每次选一个入度为0的节点开始输出，删除该节点，其连接的节点入度-1。

### 边遍历list边删除

+ 从后向前遍历即可。

### 进制转换[660.移除9](https://leetcode-cn.com/problems/remove-9/solution/yi-chu-9-by-leetcode/)

+ 删除所有带9的数字，求第n个。写出以后发现即求n的9进制数。