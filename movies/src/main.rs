use std::{env, time::Instant};
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug)]
struct Movie {
    title: String,
    description: String
}

impl Movie {
    fn new(title: String, description: String) -> Self {
        Movie { title, description }
    }
}

fn parse(filename: &str) -> Vec<Movie> {
    let file = File::open(filename).expect("Could not open {filename}");
    let reader = BufReader::new(file);

    reader
        .lines()
        .map_while(Result::ok)
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 2 {
                return None;
            }
            Some(Movie::new(parts[0].to_string(), parts[1].to_string()))
        })
        .collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        println!("Usage: ./movies <movies.txt>");
        return;
    }

    println!("Parsing {}...", args[1]);
    let s = Instant::now();
    let movies = parse(&args[1]);
    println!("Parsed in {:.2?}", s.elapsed());
}
