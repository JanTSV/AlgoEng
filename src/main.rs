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
    let graph = parse_graph("inputs/stgtregbz_ch.fmi").unwrap();
    let duration = start.elapsed();
    println!("Loaded graph in {:.2?}", duration);

    // Test Dijkstra vs CH
    let mut dijkstra = Dijkstra::new(&graph);
    const START: usize = 377371;
    const TARGET: usize = 754742;
    print!("Dijkstra: ");
    let start = Instant::now();
    let dijkstra_found = dijkstra.shortest_path(START, TARGET);
    match dijkstra_found {
        Some(dist) => print!("Found a shortest path from {START} to {TARGET}: {dist} "),
        None => print!("Did NOT find a path between {START} and {TARGET} ")
    }
    let duration = start.elapsed();
    println!("[{:.2?}]", duration);
    let mut ch = CH::new(&graph);

    print!("CH (without stall-on-demand): ");
    let start = Instant::now();
    let ch_found = ch.shortest_path(START, TARGET, false);
    match ch_found {
        Some(dist) => print!("Found a shortest path from {START} to {TARGET}: {dist} "),
        None => print!("Did NOT find a path between {START} and {TARGET} ")
    }
    let duration = start.elapsed();
    println!("[{:.2?}]", duration);

    assert_eq!(dijkstra_found, ch_found);
}