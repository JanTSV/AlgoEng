use std::fs::File;
use std::io::{BufRead, BufReader};
use std::error::Error;
use std::time::Instant;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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

fn permutate_graph(graph: &OffsetArray, perms: &Vec<u64>) -> OffsetArray {
    let mut perm_edges: Vec<Vec<Edge>> = vec![Vec::new(); perms.len()];

    for (i, perm) in perms.iter().enumerate() {
        for j in graph.1[*perm as usize]..graph.1[*perm as usize + 1] {
            let (target, weight) = graph.0[j as usize];
            perm_edges[i].push((perms[target as usize], weight));
        }
    }

    create_offset_array(perm_edges)
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct Distance {
    weight: u64,
    id: u64,
}

impl Distance {
    pub fn new(weight: u64, id: u64) -> Self {
        Distance { weight, id }
    }
}

impl Ord for Distance {
    fn cmp(&self, other: &Self) -> Ordering {
        // Flip this to convert the defaulty max heap to a min heap
        other.weight.cmp(&self.weight)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Distance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct Dijkstra<'a> {
    graph: &'a OffsetArray,
    distances: Vec<Option<u64>>,
    heap: BinaryHeap<Distance>
}

impl<'a> Dijkstra<'a> {
    pub fn new(graph: &'a OffsetArray) -> Self {
        Dijkstra { graph, distances: vec![None; graph.1.len() - 1], heap: BinaryHeap::new() }
    }

    pub fn shortest_path(&mut self, s: u64, t: u64) -> Option<u64> {
        // Push start to heap and set dist to 0
        self.heap.push(Distance::new(0, s));
        self.distances[s as usize] = Some(0);

        while let Some(Distance { weight, id }) = self.heap.pop() {
            if id == t {
                return Some(weight);
            }

            for i in self.graph.1[id as usize]..self.graph.1[id as usize + 1] {
                let edge = self.graph.0[i as usize];
                match self.distances[edge.0 as usize] {
                    None => {
                        self.distances[edge.0 as usize] = Some(weight + edge.1);
                        self.heap.push(Distance::new(weight + edge.1, edge.0));
                    }
                    Some(curr_weight) if curr_weight > weight + edge.1  => {
                        self.distances[edge.0 as usize] = Some(weight + edge.1);
                        self.heap.push(Distance::new(weight + edge.1, edge.0));
                    }
                    _ => continue
                }
            }
        }

        None
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Question 1: Load the DMI graph
    let start = Instant::now();
    println!("Started parsing...");
    let (graph, _) = parse_graph("inputs/germany.fmi")?;
    let duration = start.elapsed();
    println!("Loaded graph in {:.2?}", duration);

    // Question 2: Calculate the weaky connected components
    let start = Instant::now();
    let num_comps = calc_weakly_connected_comps(&graph);
    let duration = start.elapsed();
    println!("#weakly coupled components: {num_comps} [{:.2?}]", duration);
    //println!("{:?}", graph.1);

    // Question 3: Randomly permutate the graph using a hash function and try question 2 again
    let mut perms: Vec<u64> = (0..graph.1.len() - 1).map(|i| i as u64).collect();
    perms.reverse();
    let perm_graph = permutate_graph(&graph, &perms);
    let start = Instant::now();
    let num_comps = calc_weakly_connected_comps(&perm_graph);
    let duration = start.elapsed();
    println!("#weakly coupled components (perm): {num_comps} [{:.2?}]", duration);

    // Question 4: Shortest path of 100 randomly chosen (s, t) pairs
    let mut dijkstra = Dijkstra::new(&graph);
    let s: u64 = 8371827;
    let t: u64 = 16743653;
    let start = Instant::now();
    match dijkstra.shortest_path(s, t) {
        Some(dist) => print!("Found a shortest path from {s} to {t}: {dist}"),
        None => print!("Did NOT find a path between {s} and {t}")
    }
    let duration = start.elapsed();
    println!(" [{:.2?}]", duration);


    Ok(())
}
