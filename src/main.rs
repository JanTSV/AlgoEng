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

fn custom_rng(seed: u64) -> Box<dyn FnMut() -> u64> {
    // Simple Linear Congruential Generator (LCG): x_n+1 = (a * x_n + c) % m
    // Constants chosen for good randomness properties (from Knuth's book)
    let a: u64 = 6364136223846793005;
    let c: u64 = 1;
    let m: u64 = 2u64.pow(64);

    let mut state = seed;

    Box::new(move || {
        state = state.wrapping_mul(a).wrapping_add(c);
        state % m
    })
}

fn random_permutation_list(n: usize, seed: u64) -> Vec<u64> {
    let mut perm: Vec<u64> = (0..n as u64).collect();
    let mut rng = custom_rng(seed); // Get a custom RNG function

    // Fisher-Yates shuffle using custom RNG
    for i in (1..n).rev() {
        let j = (rng() % (i as u64 + 1)) as usize; // Generate a random index between 0 and i
        perm.swap(i, j); // Swap the elements at indices i and j
    }

    perm
}

fn permute_graph(original: &OffsetArray, perm: &[u64]) -> OffsetArray {
    let num_nodes: usize = perm.len();
    let mut adj_list = vec![Vec::new(); num_nodes];

    for node in 0..num_nodes {
        let start = original.1[node]; // Start index for node in the flat edge list
        let end = original.1[node + 1]; // End index for node in the flat edge list

        // Traverse each edge for the node
        for i in start..end {
            let (target, weight) = original.0[i as usize];

            // Apply the permutation to source and target
            let new_source = perm[node]; // Apply permutation to source node
            let new_target = perm[target as usize]; // Apply permutation to target node

            adj_list[new_source as usize].push((new_target, weight)); // Update adj list
        }
    }

    create_offset_array(adj_list) // Rebuild the offset array with permuted edges
}

fn main() -> Result<(), Box<dyn Error>> {
    // Question 1: Load the DMI graph
    let (graph, _) = parse_graph("inputs/MV.fmi")?;

    // Question 2: Calculate the weaky connected components
    let start = Instant::now();
    let num_comps = calc_weakly_connected_comps(&graph);
    let duration = start.elapsed();
    println!("#weakly coupled components: {num_comps} [{:.2?}]", duration);
    //println!("{:?}", graph.1);

    // Question 3: Randomly permutate the graph using a hash function and try question 2 again
    let salt = 0x0;
    let perm = random_permutation_list(graph.1.len() - 1, salt);
    let perm_graph = permute_graph(&graph, &perm);

    let start = Instant::now();
    let num_comps = calc_weakly_connected_comps(&perm_graph);
    let duration = start.elapsed();
    println!("#weakly coupled components (perm): {num_comps} [{:.2?}]", duration);
    //println!("{:?}", perm_graph.1);

    Ok(())
}
