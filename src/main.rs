use std::fs::File;
use std::io::{BufRead, BufReader};
use std::error::Error;

#[derive(Debug)]
struct Node {
    id: u64,
    osm_id: u64,
    lat: f64,
    lon: f64,
    height: f64,
}

#[derive(Debug)]
struct Edge {
    source: u64,
    target: u64,
    weight: u64,
    edge_type: u64,
    max_speed: u64,
}

fn parse_graph(filename: &str) -> Result<(Vec<Node>, Vec<Edge>), Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    // Filter out comments and empty lines
    let mut lines = reader
        .lines()
        .filter_map(Result::ok) 
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty() && !line.starts_with('#'));

    // Read the first 2 lines consisting of the number of nodes and the number of edges
    let num_nodes: usize = lines.next().ok_or("Missing number of nodes")?.parse()?;
    let num_edges: usize = lines.next().ok_or("Missing number of edges")?.parse()?;

    // Read the nodes
    let mut nodes = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        let line = lines.next().ok_or("Missing node line")?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        let node = Node {
            id: parts[0].parse()?,
            osm_id: parts[1].parse()?,
            lat: parts[2].parse()?,
            lon: parts[3].parse()?,
            height: parts[4].parse()?,
        };
        nodes.push(node);
    }

    // Read the edges
    let mut edges = Vec::with_capacity(num_edges);
    for _ in 0..num_edges {
        let line = lines.next().ok_or("Missing edge line")?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        let edge = Edge {
            source: parts[0].parse()?,
            target: parts[1].parse()?,
            weight: parts[2].parse()?,
            edge_type: parts[3].parse()?,
            max_speed: parts[4].parse()?,
        };
        edges.push(edge);
    }

    // TODO: Return the offset-arrays
    Ok((nodes, edges))
}


fn main() -> Result<(), Box<dyn Error>> {
    let (nodes, edges) = parse_graph("inputs/MV.fmi")?;

    Ok(())
}
