use std::{env, time::Instant};
use rand::{thread_rng, seq::IteratorRandom};

fn naive(a: &[u32], b: &[u32]) {

}

fn binary_search(a: &[u32], b: &[u32]) {

}

fn galopping_search(a: &[u32], b: &[u32]) {

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
        let mut b: Vec<u32> = (0..n).choose_multiple(&mut rng, (n / 2u32.pow(i)) as usize);
        b.sort();

        let s = Instant::now();
        naive(&a, &b);
        println!("naive() took: {:.2?}", s.elapsed());

        let s = Instant::now();
        binary_search(&a, &b);
        println!("binary_search() took: {:.2?}", s.elapsed());

        let s = Instant::now();
        galopping_search(&a, &b);
        println!("galopping_search() took: {:.2?}", s.elapsed());

    } 
}
