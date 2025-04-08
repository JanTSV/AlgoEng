use std::fs::File;
use std::io::{BufRead, BufReader};
use std::error::Error;
use std::time::Instant;

type Edge = (u64, u64);
type OffsetArray = (Vec<Edge>, Vec<u64>);

fn create_offset_array(adj_list: Vec<Vec<Edge>>) -> OffsetArray {
    let mut flat_edges = Vec::new();
    let mut offsets = Vec::with_capacity(adj_list.len() + 1);
    let mut current_offset = 0u64;

    offsets.push(current_offset);
    for edges in adj_list {
        current_offset += edges.len() as u64;
        flat_edges.extend(edges);
        offsets.push(current_offset);
    }

    (flat_edges, offsets)
}

fn parse_graph(filename: &str) -> Result<(OffsetArray, OffsetArray), Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    // Filter out comments and empty lines
    let mut lines = reader
        .lines()
        .filter_map(Result::ok)
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty() && !line.starts_with('#'));

    // Parse number of nodes and edges
    let num_nodes: u64 = lines.next().ok_or("Missing number of nodes")?.parse()?;
    let num_edges: u64 = lines.next().ok_or("Missing number of edges")?.parse()?;

    // Skip node definitions
    lines.nth((num_nodes - 1) as usize).ok_or("Missing node lines")?;

    // Build adjacency lists
    let mut outgoing_edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes as usize];
    let mut incoming_edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes as usize];

    for _ in 0..num_edges {
        let line = lines.next().ok_or("Missing edge line")?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 3 {
            return Err("Malformed edge line".into());
        }

        let source: u64 = parts[0].parse()?;
        let target: u64 = parts[1].parse()?;
        let weight: u64 = parts[2].parse()?;

        outgoing_edges[source as usize].push((target, weight));
        incoming_edges[target as usize].push((source, weight));
    }

    let outgoing = create_offset_array(outgoing_edges);
    let incoming = create_offset_array(incoming_edges);
    assert_eq!(num_nodes as usize, outgoing.1.len() - 1);
    assert_eq!(num_edges as usize, outgoing.0.len());
    Ok((outgoing, incoming))
}

fn dfs(graph: &OffsetArray, start: u64, visited: &mut Vec<bool>) {
    let mut stack = Vec::new();
    stack.push(start);

    while let Some(node) = stack.pop() {
        let node = node as usize;
        if visited[node] {
            continue;
        }

        visited[node] = true;

        for i in graph.1[node]..graph.1[node + 1] {
            let neighbor = graph.0[i as usize].0;
            if !visited[neighbor as usize] {
                stack.push(neighbor);
            }
        }
    }
}

fn calc_weakly_connected_comps(graph: &OffsetArray) -> usize {
    let mut num_comps = 0;
    let mut visited = vec![false; graph.1.len() - 1];

    for node in 0..graph.1.len() - 1 {
        if !visited[node] {
            num_comps += 1;
            dfs(graph, node as u64, &mut visited);
        }
    }

    num_comps
}

fn main() -> Result<(), Box<dyn Error>> {
    let (graph, _) = parse_graph("inputs/MV.fmi")?;

    // Calculate the weaky connected components
    let start = Instant::now();
    let num_comps = calc_weakly_connected_comps(&graph);
    let duration = start.elapsed();
    println!("#weakly coupled components: {num_comps} [{:.2?}]", duration);
    Ok(())
}
