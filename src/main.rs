use std::time::Instant;

use crate::{ch::CH, graph::Graph};

mod graph;
mod reader;
mod dijkstra;
mod ch;

fn main() {
    const GRAPH: &str = "inputs/germany.fmi";
    const QUERRIES: &str = "inputs/querries_germany.fmi";
    const OUTPUT: &str = "graph.ch";

    // Load graph
    let start = Instant::now();
    println!("Started parsing...");
    let graph = Graph::from_file(GRAPH).unwrap();
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
    
    // Querry
    for (s, t) in reader::parse_queries(QUERRIES).expect("No querries") {
        let start = Instant::now();
        let dijkstra_found = ch.shortest_path(s, t, true);
        match dijkstra_found {
            Some(dist) => print!("Found a shortest path from {s} to {t}: {dist} "),
            None => print!("Did NOT find a path between {s} and {t} ")
        }
        println!("[{}ms]", start.elapsed().as_millis());
    }

}