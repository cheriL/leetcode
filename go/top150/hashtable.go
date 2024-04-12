package top150

import (
	"sort"
	"strings"
)

func canConstruct(ransomNote string, magazine string) bool {
	charSet := map[int32]int{}
	for _, v := range magazine {
		if _, ok := charSet[v]; ok {
			charSet[v]++
		} else {
			charSet[v] = 1
		}
	}

	for _, v := range ransomNote {
		if num, ok := charSet[v]; ok && num > 0 {
			charSet[v]--
		} else {
			return false
		}
	}

	return true
}

func isIsomorphic(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}

	dic := map[uint8]uint8{}
	used := map[uint8]struct{}{}
	for i := 0; i < len(s); i++ {
		if _, ok := dic[s[i]]; ok {
			if dic[s[i]] != t[i] {
				return false
			}
		} else {
			if _, ok := used[t[i]]; ok {
				return false
			}
			dic[s[i]] = t[i]
			used[t[i]] = struct{}{}
		}
	}
	return true
}

func wordPattern(pattern string, s string) bool {
	strs := strings.Split(s, " ")
	if len(strs) != len(pattern) {
		return false
	}

	dicChar2Str := map[uint8]string{}
	dicStr2Char := map[string]uint8{}
	for idx, str := range strs {
		s, ok := dicChar2Str[pattern[idx]]
		c, ok2 := dicStr2Char[str]
		if ok && ok2 {
			if s != str || c != pattern[idx] {
				return false
			}
		} else if !ok && !ok2 {
			dicChar2Str[pattern[idx]] = str
			dicStr2Char[str] = pattern[idx]
		} else {
			return false
		}
	}

	return true
}

type myStr []rune

func newMyStr(str string) (s myStr) {
	s = make([]rune, 0, len(str))
	for _, r := range str {
		s = append(s, r)
	}
	return
}
func (s myStr) Len() int { return len(s) }
func (s myStr) Less(i int, j int) bool {
	if s[i] < s[j] {
		return true
	}
	return false
}
func (s myStr) Swap(i int, j int) {
	s[i], s[j] = s[j], s[i]
}
func (s myStr) Equal(s2 myStr) bool {
	if s.Len() != s2.Len() {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] != s2[i] {
			return false
		}
	}
	return true
}
func groupAnagrams(strs []string) (results [][]string) {
	table := map[string][]string{}
	for _, str := range strs {
		s := newMyStr(str)
		sort.Sort(s)
		if _, ok := table[string(s)]; ok {
			table[string(s)] = append(table[string(s)], str)
		} else {
			table[string(s)] = []string{str}
		}
	}
	for _, strs := range table {
		results = append(results, strs)
	}
	return
}
