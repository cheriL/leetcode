struct Solution {}

impl Solution {
  pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
    let target = &target;
    if target > &nums[nums.len() - 1] {
      return nums.len() as i32
    }

    let end = (nums.len() - 1) as i32;
    binary_search(&(0 as i32),  &end, &nums, target)
  }
}

fn binary_search(start: &i32, end: &i32, nums: &Vec<i32>, target: &i32) -> i32 {
  if start >= end {
    return *end
  }

  let s = *start as usize;
  let e = *end as usize;
  let mid = (s + e) / 2;
  let mid_num = &nums[mid];
  if mid_num == target {
    return mid as i32
  } else if mid_num < target {
    return binary_search(&(mid as i32 + 1), end, nums, target)
  } else {
    return binary_search(start, &(mid as i32), nums, target)
  }  
}