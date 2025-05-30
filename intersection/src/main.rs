use std::{env, time::Instant};
use rand::{thread_rng, seq::IteratorRandom};

fn intersect_naive(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut i_a = 0;
    let mut i_b = 0;
    let mut result= Vec::new();

    while i_a != a.len() && i_b != b.len() {
        if a[i_a] == b[i_b] {
            result.push(a[i_a]);
            i_a += 1;
            i_b += 1;
        }
        else if a[i_a] <= b[i_b] {
            i_a += 1;
        } else {
            i_b += 1;
        }
    }

    result
}

fn intersect_binary_search(a: &[u32], b: &[u32]) -> Vec<u32> {
    b.iter().filter_map(|be| if binary_search(a, *be, 0, a.len()) { Some(*be) } else { None }).collect()
}

fn binary_search(v: &[u32], s: u32, mut left: usize, mut right: usize) -> bool {
    while left < right {
        let mid = left + (right - left) / 2;
        if v[mid] == s {
            return true;
        } else if v[mid] < s {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    false
}

fn intersect_galopping_search(a: &[u32], b: &[u32]) -> Vec<u32> {
    b.iter().filter_map(|be| if galloping_search(a, *be) { Some(*be) } else { None }).collect()
}

fn galloping_search(v: &[u32], s: u32) -> bool {
    if v.is_empty() {
        return false;
    }

    let mut bound = 1;

    while bound < v.len() && v[bound] < s {
        bound *= 2;
    }

    binary_search(v, s, bound / 2, (bound + 1).min(v.len()))
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        println!("Usage: ./intersection <n>");
        return;
    }

    let n: u32 = args[1].parse().expect("Given n cannot be parsed.");

    let a: Vec<u32> = (0..n).collect();

    let mut rng = thread_rng();
    for i in 0..n.ilog2() {
        println!("-----");
        let mut b: Vec<u32> = (0..n).choose_multiple(&mut rng, (n / 2u32.pow(i)) as usize);
        b.sort();

        let s = Instant::now();
        let result_native = intersect_naive(&a, &b);
        //dbg!(&result_native);
        println!("intersect_naive() took: {:.2?}", s.elapsed());
        assert_eq!(result_native.len(), b.len());

        let s = Instant::now();
        let result_binary_search = intersect_binary_search(&a, &b);
        println!("intersect_binary_search() took: {:.2?}", s.elapsed());
        assert_eq!(result_native, result_binary_search);
        //dbg!(&result_binary_search);

        let s = Instant::now();
        let result_galloping: Vec<u32> = intersect_galopping_search(&a, &b);
        println!("intersect_galopping_search() took: {:.2?}", s.elapsed());
        assert_eq!(result_native, result_galloping);

    } 
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersect_naive_basic() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![3, 4, 5, 6, 7];
        let expected = vec![3, 4, 5];
        assert_eq!(intersect_naive(&a, &b), expected);
    }

    #[test]
    fn test_intersect_binary_search_basic() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![3, 4, 5, 6, 7];
        let expected = vec![3, 4, 5];
        assert_eq!(intersect_binary_search(&a, &b), expected);
    }

    #[test]
    fn test_intersect_galloping_search_basic() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![3, 4, 5, 6, 7];
        let expected = vec![3, 4, 5];
        assert_eq!(intersect_galopping_search(&a, &b), expected);
    }

    #[test]
    fn test_no_intersection() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let expected: Vec<u32> = vec![];
        assert_eq!(intersect_naive(&a, &b), expected);
        assert_eq!(intersect_binary_search(&a, &b), expected);
        assert_eq!(intersect_galopping_search(&a, &b), expected);
    }

    #[test]
    fn test_empty_inputs() {
        let a: Vec<u32> = vec![];
        let b: Vec<u32> = vec![];
        let expected: Vec<u32> = vec![];
        assert_eq!(intersect_naive(&a, &b), expected);
        assert_eq!(intersect_binary_search(&a, &b), expected);
        assert_eq!(intersect_galopping_search(&a, &b), expected);
    }
}