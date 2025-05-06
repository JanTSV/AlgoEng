use std::{time::Instant, fs::OpenOptions, io::Write};

use crate::{ch::CH, graph::Graph};

mod graph;
mod reader;
mod dijkstra;
mod ch;

fn main() {
    // Inputs
    const GRAPH: &str = "inputs/germany.fmi";
    const QUERRIES: &str = "inputs/querries_germany.txt";
    const OUTPUT: &str = "graph.ch";

    // Outputs
    const LOG: &str = "log.txt";
    const RESULT: &str = "results.txt";

    let mut log = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(LOG).unwrap();

    let mut result = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(RESULT).unwrap();

    // Load graph
    let start = Instant::now();
    writeln!(log, "Started parsing...").unwrap();
    let graph = Graph::from_file(GRAPH).unwrap();
    let duration = start.elapsed();
    writeln!(log, "Loaded graph in {:.2?}", duration).unwrap();

    // Preprocessing
    let mut ch = CH::new(graph);
    writeln!(log, "#original edges: {}", ch.get_graph().num_edges()).unwrap();

    let start = Instant::now();
    writeln!(log, "Started CH preprocessing...").unwrap();
    let num_shortcuts = ch.batch_preprocess();
    let duration = start.elapsed();
    writeln!(log, "Preprocessed in {:.2?}", duration).unwrap();
    writeln!(log, "#created edges {}, #edges in new graph: {}", num_shortcuts, ch.get_graph().num_edges()).unwrap();

    // Print out the graph
    let start = Instant::now();
    writeln!(log, "Writing graph to file...").unwrap();
    // ch.get_graph().to_file(OUTPUT, GRAPH).unwrap();
    let duration = start.elapsed();
    writeln!(log, "Written in {:.2?}", duration).unwrap();

    // Querry
    for (s, t) in reader::parse_queries(QUERRIES).expect("No querries") {
        let start = Instant::now();
        let dijkstra_found = ch.shortest_path(s, t, true);
        let duration = start.elapsed().as_micros();
        match dijkstra_found {
            Some(dist) => writeln!(result, "{} {} {} {}", s, t, dist, duration).unwrap(),
            None => writeln!(result, "{} {} INF {}", s, t, duration).unwrap()
        }
    }
}