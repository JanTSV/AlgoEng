use std::fs::File;
use std::io::{BufRead, BufReader};
use std::error::Error;

type Edge = (u64, u64, u64);
type OffsetArray = (Vec<Edge>, Vec<u64>);

fn parse_graph(filename: &str) -> Result<(OffsetArray, OffsetArray), Box<dyn Error>> {
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

    // Risky assumption: all nodes are defined in order and gap less
    lines.nth((num_nodes - 1) as usize).ok_or("Missing node lines")?;

    // Read the edges
    let mut outgoing_edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes as usize];
    let mut incoming_edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes as usize];
    for _ in 0..num_edges {
        let line = lines.next().ok_or("Missing edge line")?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        let source : u64 = parts[0].parse()?;
        let target : u64 = parts[1].parse()?;
        let weight : u64 = parts[2].parse()?;
        outgoing_edges[source as usize].push((source, target, weight));
        incoming_edges[target as usize].push((source, target, weight));
    }

    // Return the offset-arrays
    // Following data structures are needed:
    // Edges:                   Array-Offsets
    // (src, target, distance)  (Node, EndIdx)
    // (0, 1, 2)                (0, 2)
    // (0, 3, 12)               (1, 4 -> from 2)
    // (0, 15, 78)              (2, 4)
    // (2, 3, 5)                ...
    // (2, 89, 1)               (N, M)
    // ....
    let mut flat_outgoing_edges: Vec<Edge> = Vec::with_capacity(num_edges as usize);
    let mut outgoing_offsets: Vec<u64> = Vec::with_capacity(num_nodes as usize + 1);
    
    let mut current_offset = 0;
    
    for edges_for_node in outgoing_edges {
        current_offset += edges_for_node.len() as u64;
        flat_outgoing_edges.extend(edges_for_node);
        outgoing_offsets.push(current_offset);
    }
    outgoing_offsets.push(current_offset);

    let mut flat_incoming_edges: Vec<Edge> = Vec::with_capacity(num_edges as usize);
    let mut incoming_offsets: Vec<u64> = Vec::with_capacity(num_nodes as usize + 1);
    
    let mut current_offset = 0;
    
    for edges_for_node in incoming_edges {
        current_offset += edges_for_node.len() as u64;
        flat_incoming_edges.extend(edges_for_node);
        incoming_offsets.push(current_offset);
    }
    incoming_offsets.push(current_offset);


    Ok(((flat_outgoing_edges, outgoing_offsets), (flat_incoming_edges, incoming_offsets)))
}


fn main() -> Result<(), Box<dyn Error>> {
    parse_graph("inputs/MV.fmi")?;

    Ok(())
}
