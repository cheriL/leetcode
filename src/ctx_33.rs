pub struct Solution {}

impl Solution {
  pub fn search(nums: Vec<i32>, target: i32) -> i32 {
    let spin_index = index_search(&nums);
    let mut res = binary_search(&0, &(spin_index), &target, &nums);
    if res < 0 {
      res = binary_search(&spin_index, &(nums.len() as i32), &target, &nums);
    }
    
    res
  }
}

//先找到旋转点
fn index_search<T: std::cmp::PartialOrd>(nums: &Vec<T>) -> i32 {
  let mut left = 0;
  let mut right = (nums.len() - 1) as i32;
  // if nums.len() <= 0 || nums[0] <= nums[right as usize] {
  //   return 0
  // }
  let mut res = 0;
  
  while left < right {
    let mid = (left + right) / 2;
    if nums.get(left as usize) > nums.get(right as usize) {
      res = right;
    }
    if nums.get(left as usize) > nums.get(mid as usize) {
      res = mid;
      right = mid - 1;
    } else if left == mid {
      left = mid + 1;
    } else {
      left = mid
    }
  }

  res
}

fn binary_search<T: std::cmp::PartialOrd>(start: &i32, end: &i32, target: &T, nums: &Vec<T>) -> i32 {
  let mut res = -1;
  let mut left = start.clone();
  let mut right = end.clone() - 1;

  while left <= right {
    let mid = (left + right) / 2;
    if &nums[mid as usize] == target {
      res = mid;
      break;
    }

    if &nums[mid as usize] > target {
      right = mid - 1;
    } else {
      left = mid + 1;
    }

  }

  res
}