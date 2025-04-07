use std::fs::File;
use std::io::{BufRead, BufReader};
use std::error::Error;

fn parse_graph(filename: &str) -> Result<(), Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    // Filter out comments and empty lines
    let mut lines = reader
        .lines()
        .filter_map(Result::ok) 
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty() && !line.starts_with('#'));

    // Read the first 2 lines consisting of the number of nodes and the number of edges
    let num_nodes: u64 = lines.next().ok_or("Missing number of nodes")?.parse()?;
    let num_edges: u64 = lines.next().ok_or("Missing number of edges")?.parse()?;

    // Read the nodes
    for _ in 0..num_nodes {
        let line = lines.next().ok_or("Missing node line")?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        let id: u64 = parts[0].parse()?;
        assert!(id < num_nodes, "Node ID is bigger than num_nodes");
        // let _osm_id : u64 = parts[1].parse()?;
        // let _lat : f64 = parts[2].parse()?;
        // let _lon : f64 = parts[3].parse()?;
        // let _height : u64 = parts[4].parse()?;
    }

    // Read the edges
    for _ in 0..num_edges {
        let line = lines.next().ok_or("Missing edge line")?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        let source : u64 = parts[0].parse()?;
        let target : u64 = parts[1].parse()?;
        let weight : u64 = parts[2].parse()?;
        // let _edge_type : u64 = parts[3].parse()?;
        // let _max_speed : u64 = parts[4].parse()?;
    }

    // TODO: Return the offset-arrays
    // Following data structures are needed:
    // Edges:                   Array-Offsets
    // (src, target, distance)  (Node, EndIdx)
    // (0, 1, 2)                (0, 2)
    // (0, 3, 12)               (1, 4 -> from 2)
    // (0, 15, 78)              (2, 4)
    // (2, 3, 5)                ...
    // (2, 89, 1)               (N, M)
    // ....
    Ok(())
}


fn main() -> Result<(), Box<dyn Error>> {
    parse_graph("inputs/MV.fmi")?;

    Ok(())
}
