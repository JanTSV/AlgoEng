use std::{env, time::Instant};
use rand::{thread_rng, Rng};

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
    let mut left = 0;
    b
        .iter()
        .filter_map(|be| {
            if let Some(new_left) = binary_search(a, *be, left, a.len()) {
                left = new_left;
                Some(*be) 
            } else { 
                None 
            }
        })
        .collect()
}

fn binary_search(v: &[u32], s: u32, mut left: usize, mut right: usize) -> Option<usize> {
    while left < right {
        let mid = left + (right - left) / 2;
        if v[mid] == s {
            return Some(mid);
        } else if v[mid] < s {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    None
}

fn intersect_galopping_search(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut left = 0;
    b
        .iter()
        .filter_map(|be| {
            if let Some(new_left) = galloping_search(a, *be, left) {
                left = new_left;
                Some(*be) 
            } else { 
                None 
            }
        })
        .collect()
}

fn galloping_search(v: &[u32], s: u32, left: usize) -> Option<usize> {
    if v.is_empty() {
        return None;
    }

    let mut bound = left + 1;

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
        let mut b: Vec<u32> = (0..(n / 2_u32.pow(i))).map(|_| rng.gen_range(0..n)).collect();
        b.sort();
        b.dedup();
        println!("|a| = {}, |b| = {}", a.len(), b.len());


        let s = Instant::now();
        let result_naive = intersect_naive(&a, &b);
        //dbg!(&result_native);
        println!("intersect_naive() took: {:.2?}", s.elapsed());
        assert_eq!(result_naive.len(), b.len());

        let s = Instant::now();
        let result_binary_search = intersect_binary_search(&a, &b);
        println!("intersect_binary_search() took: {:.2?}", s.elapsed());
        assert_eq!(result_naive, result_binary_search);
        //dbg!(&result_binary_search);

        let s = Instant::now();
        let result_galloping: Vec<u32> = intersect_galopping_search(&a, &b);
        println!("intersect_galopping_search() took: {:.2?}", s.elapsed());
        assert_eq!(result_naive, result_galloping);

    } 
}

#[cfg(test)]
mod tests {
    use rand::seq::IteratorRandom;

    use super::*;
    use std::fs::OpenOptions;
    use std::io::{self, Write};

    #[test]
    fn test_whats_faster() {
        const n: u32 = 100000000;

        #[derive(Debug, PartialEq)]
        enum IntersectionType {
            NONE,
            NAIVE,
            BINARY,
            GALLOPING
        }

        let a: Vec<u32> = (0..n).collect();
        let mut fastest = IntersectionType::NONE;
    
        let mut rng = thread_rng();
        for i in 0..n.ilog2() {
            let mut b: Vec<u32> = (0..n).choose_multiple(&mut rng, (n / 2u32.pow(i)) as usize);
            b.sort();
    
            let s = Instant::now();
            let result_naive = intersect_naive(&a, &b);
            let duration_naive = s.elapsed();
            assert_eq!(result_naive.len(), b.len());
    
            let s = Instant::now();
            let result_binary_search = intersect_binary_search(&a, &b);
            let duration_binary = s.elapsed();
            assert_eq!(result_naive, result_binary_search);
    
            let s = Instant::now();
            let result_galloping: Vec<u32> = intersect_galopping_search(&a, &b);
            let duration_galloping = s.elapsed();
            assert_eq!(result_naive, result_galloping);
            
            if duration_naive < duration_binary && duration_naive < duration_galloping && fastest != IntersectionType::NAIVE {
                fastest = IntersectionType::NAIVE;
                println!("-----");
                println!("Fastest for |a| = {} and |b| = {}: {:?} [{:.2?}].", a.len(), b.len(), fastest, duration_naive);
                println!("Others: {:?} [{:.2?}], {:?} [{:.2?}]", IntersectionType::BINARY, duration_binary, IntersectionType::GALLOPING, duration_galloping);
            } else if duration_binary < duration_naive && duration_binary < duration_galloping && fastest != IntersectionType::BINARY {
                fastest = IntersectionType::BINARY;
                println!("-----");
                println!("Fastest for |a| = {} and |b| = {}: {:?} [{:.2?}].", a.len(), b.len(), fastest, duration_binary);
                println!("Others: {:?} [{:.2?}], {:?} [{:.2?}]", IntersectionType::NAIVE, duration_naive, IntersectionType::GALLOPING, duration_galloping);
            } else if duration_galloping < duration_naive && duration_galloping < duration_binary && fastest != IntersectionType::GALLOPING {
                fastest = IntersectionType::GALLOPING;
                println!("-----");
                println!("Fastest for |a| = {} and |b| = {}: {:?} [{:.2?}].", a.len(), b.len(), fastest, duration_galloping);
                println!("Others: {:?} [{:.2?}], {:?} [{:.2?}]", IntersectionType::NAIVE, duration_naive, IntersectionType::BINARY, duration_binary);
            }
        } 
    }

    #[test]
    fn csv_test_whats_faster() {
        const n: u32 = 100000000;
        let a: Vec<u32> = (0..n).collect();
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open("csv_test_whats_faster.csv")
            .expect("Cannot create file");
    
        writeln!(file, "ab,duration_naive,duration_binary,duration_galloping").unwrap();

        let mut rng = thread_rng();
        for i in 0..n.ilog2() {
            let mut b: Vec<u32> = (0..n).choose_multiple(&mut rng, (n / 2u32.pow(i)) as usize);
            b.sort();
    
            let s = Instant::now();
            let result_naive = intersect_naive(&a, &b);
            let duration_naive = s.elapsed();
            assert_eq!(result_naive.len(), b.len());
    
            let s = Instant::now();
            let result_binary_search = intersect_binary_search(&a, &b);
            let duration_binary = s.elapsed();
            assert_eq!(result_naive, result_binary_search);
    
            let s = Instant::now();
            let result_galloping: Vec<u32> = intersect_galopping_search(&a, &b);
            let duration_galloping = s.elapsed();
            assert_eq!(result_naive, result_galloping);

            writeln!(file, "{},{},{},{}", (a.len() as f64 / b.len() as f64).ln(), duration_naive.as_nanos(), duration_binary.as_nanos(), duration_galloping.as_nanos()).unwrap();
        } 
    }

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