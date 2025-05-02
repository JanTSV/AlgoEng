use std::time::Instant;

use crate::{reader::parse_graph, dijkstra::Dijkstra, ch::CH};

mod graph;
mod reader;
mod dijkstra;
mod ch;
mod perm;

fn main() {
    // Load graph with CH levels
    let start = Instant::now();
    println!("Started parsing...");
    let graph = parse_graph("inputs/germany.fmi").unwrap();
    let duration = start.elapsed();
    println!("Loaded graph in {:.2?}", duration);

    // Test Dijkstra vs CH
    let mut dijkstra = Dijkstra::new(&graph);
    const START: usize = 8371825;
    const TARGET: usize = 16743651;
    print!("Dijkstra: ");
    let start = Instant::now();
    let dijkstra_found = dijkstra.shortest_path(START, TARGET);
    match dijkstra_found {
        Some(dist) => print!("Found a shortest path from {START} to {TARGET}: {dist} "),
        None => print!("Did NOT find a path between {START} and {TARGET} ")
    }
    let duration = start.elapsed();
    println!("[{:.2?}]", duration);
}