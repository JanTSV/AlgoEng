use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
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
        .map_while(Result::ok)
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty() && !line.starts_with('#'));

    // Parse number of nodes and edges
    let num_nodes: u64 = lines.next().ok_or("Missing number of nodes")?.parse()?;
    let num_edges: u64 = lines.next().ok_or("Missing number of edges")?.parse()?;

    // Parse nodes
    for _ in 0..num_nodes {
        let line = lines.next().ok_or("Missing edge line")?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.len() != 6 {
            return Err("Malformed node line".into());
        }

        // TODO
        let level: u64 = parts[5].parse()?;
    }
    

    // Build adjacency lists
    let mut outgoing_edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes as usize];
    let mut incoming_edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes as usize];

    for _ in 0..num_edges {
        let line = lines.next().ok_or("Missing edge line")?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() != 7 {
            return Err("Malformed edge line".into());
        }

        let source: u64 = parts[0].parse()?;
        let target: u64 = parts[1].parse()?;
        let weight: u64 = parts[2].parse()?;
        let edge_id_a: i64 = parts[5].parse()?;
        let edge_id_b: i64 = parts[6].parse()?;

        outgoing_edges[source as usize].push((target, weight));
        incoming_edges[target as usize].push((source, weight));
    }

    let outgoing = create_offset_array(outgoing_edges);
    let incoming = create_offset_array(incoming_edges);
    assert_eq!(num_nodes as usize, outgoing.1.len() - 1);
    assert_eq!(num_edges as usize, outgoing.0.len());
    Ok((outgoing, incoming))
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
    heap: BinaryHeap<Distance>,
    visited: Vec<u64>
}

impl<'a> Dijkstra<'a> {
    pub fn new(graph: &'a OffsetArray) -> Self {
        Dijkstra { graph, distances: vec![None; graph.1.len() - 1], heap: BinaryHeap::new(), visited: Vec::new() }
    }

    pub fn shortest_path(&mut self, s: u64, t: u64) -> Option<u64> {
        // Cleaup of previous run
        while let Some(node) = self.visited.pop() {
            self.distances[node as usize] = None;
        }
        self.heap.clear();

        // Push start to heap and set dist to 0
        self.heap.push(Distance::new(0, s));
        self.distances[s as usize] = Some(0);
        self.visited.push(s);

        while let Some(Distance { weight, id }) = self.heap.pop() {
            if id == t {
                return Some(weight);
            }

            for i in self.graph.1[id as usize]..self.graph.1[id as usize + 1] {
                let edge = self.graph.0[i as usize];
                if self.distances[edge.0 as usize].is_none_or(|curr| weight + edge.1 < curr) {
                    self.distances[edge.0 as usize] = Some(weight + edge.1);
                    self.heap.push(Distance::new(weight + edge.1, edge.0));
                    self.visited.push(edge.0);
                }
            }
        }

        None
    }
}

fn read_query(filename: &str) -> Result<Vec<(u64, u64)>, Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    Ok(reader
        .lines()
        .map_while(Result::ok)
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                return None;
            }

            let source: u64 = parts[0].parse().ok()?;
            let target: u64 = parts[1].parse().ok()?;

            Some((source, target))
        })
        .collect())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Question 1

    // Load graph with CH levels
    let args: Vec<String> = env::args().collect();

    let mut log = BufWriter::new(File::create("dump.txt").expect("Could not create log"));

    let start = Instant::now();
    println!("Started parsing...");
    let (graph, incoming_graph) = parse_graph("stgtregbz_ch.fmi")?;
    let duration = start.elapsed();
    println!("Loaded graph in {:.2?}", duration);


    Ok(())
}
