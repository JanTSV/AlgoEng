use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::error::Error;
use std::time::Instant;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone)]
struct Edge {
    to: u64,
    weight: u64,
    edge_id_a: Option<u64>,
    edge_id_b: Option<u64>
}

impl Edge {
    pub fn new(to: u64, weight: u64, edge_id_a: Option<u64>, edge_id_b: Option<u64>) -> Self {
        Edge { to, weight, edge_id_a, edge_id_b}
    } 
}

#[derive(Clone)]
struct Node {
    offset: u64,
    level: u64
}

impl Node {
    pub fn new(offset: u64, level: u64) -> Self {
        Node { offset, level }
    }
}

type OffsetArray = (Vec<Edge>, Vec<Node>);

fn create_offset_array(adj_list: Vec<Vec<Edge>>, levels: &Vec<u64>) -> OffsetArray {
    let mut flat_edges: Vec<Edge> = Vec::new();
    let mut nodes: Vec<Node> = Vec::with_capacity(adj_list.len() + 1);
    let mut current_offset = 0u64;

    nodes.push(Node::new(current_offset, 0));
    for (i, edges) in adj_list.iter().enumerate() {
        current_offset += edges.len() as u64;
        flat_edges.extend(edges.clone());
        nodes.push(Node::new(current_offset, levels[i]));
    }

    (flat_edges, nodes)
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
    let mut levels: Vec<u64> = vec![];
    for _ in 0..num_nodes {
        let line = lines.next().ok_or("Missing edge line")?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.len() != 6 {
            return Err("Malformed node line".into());
        }

        let level: u64 = parts[5].parse()?;
        levels.push(level);
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
        let edge_id_a: Option<u64> = parts[5].parse().ok();
        let edge_id_b: Option<u64> = parts[6].parse().ok();

        outgoing_edges[source as usize].push(Edge::new(target, weight, edge_id_a, edge_id_b));
        incoming_edges[target as usize].push(Edge::new(source, weight, edge_id_b, edge_id_a));
    }

    let outgoing = create_offset_array(outgoing_edges, &levels);
    let incoming = create_offset_array(incoming_edges, &levels);
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

struct CH<'a> {
    // s -> t
    graph: &'a OffsetArray,
    distances: Vec<Option<u64>>,
    heap: BinaryHeap<Distance>,
    visited: Vec<u64>,

    // t -> s
    incoming_graph: &'a OffsetArray,
    incoming_distances: Vec<Option<u64>>,
    incoming_heap: BinaryHeap<Distance>,
    incoming_visited: Vec<u64>
}

impl<'a> CH<'a> {
    pub fn new(graph: &'a OffsetArray, incoming_graph: &'a OffsetArray) -> Self {
        CH { graph, 
             distances: vec![None; graph.1.len() - 1], 
             heap: BinaryHeap::new(), 
             visited: Vec::new(),
             incoming_graph,
             incoming_distances: vec![None; incoming_graph.1.len() - 1], 
             incoming_heap: BinaryHeap::new(), 
             incoming_visited: Vec::new()
        }
    }

    pub fn shortest_path(&mut self, s: u64, t: u64) -> Option<u64> {
        // Cleanup of previous run
        while let Some(node) = self.visited.pop() {
            self.distances[node as usize] = None;
        }

        while let Some(node) = self.incoming_visited.pop() {
            self.incoming_distances[node as usize] = None;
        }
        self.heap.clear();
        self.incoming_heap.clear();

        while !self.heap.is_empty() || !self.incoming_heap.is_empty() {
            if let Some(Distance { weight, id }) = self.heap.pop() {
                if id == t {
                    return Some(weight);
                }

                for i in self.graph.1[id as usize].offset..self.graph.1[id as usize + 1].offset {
                    let edge = &self.graph.0[i as usize];
                    if self.distances[edge.to as usize].is_none_or(|curr| weight + edge.weight < curr) {
                        self.distances[edge.to as usize] = Some(weight + edge.weight);
                        self.heap.push(Distance::new(weight + edge.weight, edge.to));
                        self.visited.push(edge.to);
                    }
                }
            }

            if let Some(Distance { weight, id }) = self.incoming_heap.pop() {
                if id == t {
                    return Some(weight);
                }

                for i in self.graph.1[id as usize].offset..self.graph.1[id as usize + 1].offset {
                    let edge = &self.graph.0[i as usize];
                    if self.distances[edge.to as usize].is_none_or(|curr| weight + edge.weight < curr) {
                        self.distances[edge.to as usize] = Some(weight + edge.weight);
                        self.heap.push(Distance::new(weight + edge.weight, edge.to));
                        self.visited.push(edge.to);
                    }
                }
            }
        }
        None
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
        // Cleanup of previous run
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

            for i in self.graph.1[id as usize].offset..self.graph.1[id as usize + 1].offset {
                let edge = &self.graph.0[i as usize];
                if self.distances[edge.to as usize].is_none_or(|curr| weight + edge.weight < curr) {
                    self.distances[edge.to as usize] = Some(weight + edge.weight);
                    self.heap.push(Distance::new(weight + edge.weight, edge.to));
                    self.visited.push(edge.to);
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

    let mut dijkstra = Dijkstra::new(&graph);
    let s = 377371;
    let t = 754742;
    match dijkstra.shortest_path(s, t) {
        Some(dist) => print!("Found a shortest path from {s} to {t}: {dist}"),
        None => print!("Did NOT find a path between {s} and {t}")
    }

    Ok(())
}
