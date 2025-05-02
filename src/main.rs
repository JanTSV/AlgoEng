use std::time::Instant;

use crate::{reader::parse_graph, dijkstra::Dijkstra, ch::CH, perm::Permutation};

mod graph;
mod reader;
mod dijkstra;
mod ch;
mod perm;

fn main() {
    // Load graph with CH levels
    let start = Instant::now();
    println!("Started parsing...");
    let mut graph = parse_graph("inputs/MV.fmi").unwrap();
    let duration = start.elapsed();
    println!("Loaded graph in {:.2?}", duration);

    // Dijkstra
    let mut dijkstra = Dijkstra::new(&graph);
    const START: usize = 214733;
    const TARGET: usize = 429466;
    print!("Dijkstra: ");
    let start = Instant::now();
    let dijkstra_found = dijkstra.shortest_path(START, TARGET);
    match dijkstra_found {
        Some(dist) => print!("Found a shortest path from {START} to {TARGET}: {dist} "),
        None => print!("Did NOT find a path between {START} and {TARGET} ")
    }
    let duration = start.elapsed();
    println!("[{:.2?}]", duration);

    // Preprocessing
    let mut ch = CH::new(graph);

    let start = Instant::now();
    println!("Started CH preprocessing...");
    ch.batch_preprocess();
    let duration = start.elapsed();
    println!("Preprocessed in {:.2?}", duration);

    // TODO: maybe permutate -> sort by level
    
    // CH
    print!("CH: ");
    let start = Instant::now();
    let dijkstra_found = ch.shortest_path(START, TARGET, true);
    match dijkstra_found {
        Some(dist) => print!("Found a shortest path from {START} to {TARGET}: {dist} "),
        None => print!("Did NOT find a path between {START} and {TARGET} ")
    }
    let duration = start.elapsed();
    println!("[{:.2?}]", duration);

}