package top150

func letterCombinations(digits string) (results []string) {
	if len(digits) == 0 {
		return []string{}
	}

	dic := map[uint8][]rune{
		'2': {'a', 'b', 'c'},
		'3': {'d', 'e', 'f'},
		'4': {'g', 'h', 'i'},
		'5': {'j', 'k', 'l'},
		'6': {'m', 'n', 'o'},
		'7': {'p', 'q', 'r', 's'},
		'8': {'t', 'u', 'v'},
		'9': {'w', 'x', 'y', 'z'},
	}

	var path []rune
	var backTrack func(string, []rune, *[]string, int)
	backTrack = func(s string, path []rune, results *[]string, idx int) {
		if len(path) == len(s) {
			*results = append(*results, string(path))
			return
		}

		digit := s[idx]
		for i := 0; i < len(dic[digit]); i++ {
			path = append(path, dic[digit][i])
			backTrack(s, path, results, idx+1)
			path = path[:len(path)-1]
		}
	}
	backTrack(digits, path, &results, 0)
	return results
}

func permute(nums []int) (results [][]int) {
	path := make([]int, 0, len(nums))
	used := map[int]struct{}{}

	var backTrack func([]int, []int, int, *[][]int)
	backTrack = func(nums []int, path []int, idx int, results *[][]int) {
		if len(path) == len(nums) {
			result := make([]int, len(path))
			copy(result, path)
			*results = append(*results, result)
			return
		}

		for i := 0; i < len(nums); i++ {
			if _, ok := used[i]; ok {
				continue
			}
			path = append(path, nums[i])
			used[i] = struct{}{}
			backTrack(nums, path, idx+1, results)
			delete(used, i)
			path = path[:len(path)-1]
		}
	}

	backTrack(nums, path, 0, &results)
	return
}
