use std::time::Instant;

use crate::{ch::CH, graph::Graph};

mod graph;
mod reader;
mod dijkstra;
mod ch;

fn main() {
    // Load graph
    let start = Instant::now();
    println!("Started parsing...");
    let graph = Graph::from_file("inputs/germany.fmi").unwrap();
    let duration = start.elapsed();
    println!("Loaded graph in {:.2?}", duration);

    // Preprocessing
    let mut ch = CH::new(graph);

    let start = Instant::now();
    println!("Started CH preprocessing...");
    ch.batch_preprocess();
    let duration = start.elapsed();
    println!("Preprocessed in {:.2?}", duration);

    // Print out the graph
    let start = Instant::now();
    println!("Writing graph to file...");
    // ch.get_graph().to_file("graph.ch").unwrap();
    let duration = start.elapsed();
    println!("Written in {:.2?}", duration);

    // TODO: maybe permutate -> sort by level
    
    // TODO: querry

    // CH
    const START: usize = 214733;
    const TARGET: usize = 429466;
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