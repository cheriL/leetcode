pub struct Solution {}

impl Solution {
  pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let len = nums.len() as i32;
    let left = binary_search(&0, &len, &target, &nums, true);
    let right = binary_search(&0, &len, &target, &nums, false);
    vec!(left, right)
  }
}

// [5,7,7,8,8,10]
// 8

fn binary_search(start: &i32, end: &i32, target: &i32, nums: &Vec<i32>, lower: bool) -> i32 {
  let mut left = start.clone();
  let mut right = end.clone() - 1;
  let mut res = -1;

  while left <= right {
    let mid = (left + right) / 2;
    let mid_num = &nums[mid as usize];
    if mid_num == target {
      res = mid;
      if lower {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    } else if mid_num < target {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  res
}