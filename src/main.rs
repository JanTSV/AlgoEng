use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::error::Error;
use std::process::exit;
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
type Level = (u64, u64);

fn create_offset_array(adj_list: Vec<Vec<Edge>>, levels: &Vec<Level>) -> OffsetArray {
    let mut flat_edges: Vec<Edge> = Vec::new();
    let mut nodes: Vec<Node> = Vec::with_capacity(adj_list.len() + 1);
    let mut current_offset = 0u64;

    nodes.push(Node::new(current_offset, levels[0].1));
    assert_eq!(adj_list.len(), levels.len());
    for (i, edges) in adj_list.iter().enumerate() {
        current_offset += edges.len() as u64;
        flat_edges.extend(edges.clone());
        nodes.push(Node::new(current_offset, levels.get(i + 1).map(|lvl| lvl.1).unwrap_or(0)));
    }

    (flat_edges, nodes)
}

fn parse_graph(filename: &str) -> Result<(Vec<u64>, OffsetArray, OffsetArray), Box<dyn Error>> {
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
    let mut levels: Vec<Level> = vec![];
    for _i in 0..num_nodes {
        let line = lines.next().ok_or("Missing edge line")?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.len() != 6 {
            return Err("Malformed node line".into());
        }

        let id: u64 = parts[0].parse()?;
        assert_eq!(id, _i);
        let level: u64 = parts[5].parse()?;
        levels.push((id, level));
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

        let source: u64 = levels[parts[0].parse::<usize>()?].0;
        let target: u64 = levels[parts[1].parse::<usize>()?].0;
        let weight: u64 = levels[parts[2].parse::<usize>()?].0;
        let edge_id_a: Option<u64> = parts[5].parse().ok();
        let edge_id_b: Option<u64> = parts[6].parse().ok();

        outgoing_edges[source as usize].push(Edge::new(target, weight, edge_id_a, edge_id_b));
        incoming_edges[target as usize].push(Edge::new(source, weight, edge_id_b, edge_id_a));
    }

    // Sort nodes by level
    levels.sort_by_key(|l| l.1);

    // 1. Map original node IDs to new indices after sorting
    let mut id_to_new_index = vec![0u64; levels.len()];
    for (new_index, (old_id, _)) in levels.iter().enumerate() {
        id_to_new_index[*old_id as usize] = new_index as u64;
    }

    // 2. Remap and reorder adjacency lists
    let mut sorted_outgoing = vec![Vec::new(); levels.len()];
    let mut sorted_incoming = vec![Vec::new(); levels.len()];
    for (new_index, (old_id, _)) in levels.iter().enumerate() {
        sorted_outgoing[new_index] = outgoing_edges[*old_id as usize]
            .iter()
            .map(|edge| {
                let mut new_edge = edge.clone();
                new_edge.to = id_to_new_index[edge.to as usize];
                new_edge
            })
            .collect();

        sorted_incoming[new_index] = incoming_edges[*old_id as usize]
            .iter()
            .map(|edge| {
                let mut new_edge = edge.clone();
                new_edge.to = id_to_new_index[edge.to as usize];
                new_edge
            })
            .collect();
    }

    let outgoing = create_offset_array(sorted_outgoing, &levels);
    let incoming = create_offset_array(sorted_incoming, &levels);
    assert_eq!(num_nodes as usize, outgoing.1.len() - 1);
    assert_eq!(num_edges as usize, outgoing.0.len());
    Ok((id_to_new_index, outgoing, incoming))
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
    incoming_heap: BinaryHeap<Distance>
}

impl<'a> CH<'a> {
    pub fn new(graph: &'a OffsetArray, incoming_graph: &'a OffsetArray) -> Self {
        CH { graph, 
             distances: vec![None; graph.1.len() - 1], 
             heap: BinaryHeap::new(), 
             visited: Vec::new(),
             incoming_graph,
             incoming_distances: vec![None; incoming_graph.1.len() - 1], 
             incoming_heap: BinaryHeap::new()
        }
    }

    pub fn shortest_path(&mut self, s: u64, t: u64) -> Option<u64> {
        // Cleanup of previous run
        while let Some(node) = self.visited.pop() {
            self.distances[node as usize] = None;
            self.incoming_distances[node as usize] = None;
        }

        self.heap.clear();
        self.incoming_heap.clear();

        // Push s and t to heaps and set dists to 0
        self.heap.push(Distance::new(0, s));
        self.distances[s as usize] = Some(0);
        self.visited.push(s);

        self.incoming_heap.push(Distance::new(0, t));
        self.incoming_distances[t as usize] = Some(0);
        self.visited.push(t);

        while !self.heap.is_empty() || !self.incoming_heap.is_empty() {
            // Dijkstra from s
            if let Some(Distance { weight, id }) = self.heap.pop() {
                for i in self.graph.1[id as usize].offset..self.graph.1[id as usize + 1].offset {
                    let edge = &self.graph.0[i as usize];
                    // println!("CURRENT: {} {} EDGE: {} {}", id, self.graph.1[id as usize].level, edge.to, self.graph.1[edge.to as usize].level);
                    if self.distances[edge.to as usize].is_none_or(|curr| weight + edge.weight < curr) && 
                       self.graph.1[edge.to as usize].level >= self.graph.1[id as usize].level {
                        self.distances[edge.to as usize] = Some(weight + edge.weight);
                        self.heap.push(Distance::new(weight + edge.weight, edge.to));
                        self.visited.push(edge.to);
                    }
                }
            }

            // Dijkstra from t
            if let Some(Distance { weight, id }) = self.incoming_heap.pop() {
                for i in self.incoming_graph.1[id as usize].offset..self.incoming_graph.1[id as usize + 1].offset {
                    let edge = &self.incoming_graph.0[i as usize];
                    if self.incoming_distances[edge.to as usize].is_none_or(|curr| weight + edge.weight < curr) && 
                       self.incoming_graph.1[edge.to as usize].level >= self.incoming_graph.1[id as usize].level {
                        self.incoming_distances[edge.to as usize] = Some(weight + edge.weight);
                        self.incoming_heap.push(Distance::new(weight + edge.weight, edge.to));
                        self.visited.push(edge.to);
                    }
                }
            }
        }

        // Get minimal connecting node
        let mut distance: Option<u64> = None;
        for v in &self.visited {
            match (distance, self.distances[*v as usize], self.incoming_distances[*v as usize]) {
                (None, Some(d0), Some(d1)) => distance = Some(d0 + d1),
                (Some(d), Some(d0), Some(d1)) if d0 + d1 < d => distance = Some(d0 + d1),
                _ => continue
            }
        }

        distance
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
    let mut log = BufWriter::new(File::create("dump.txt").expect("Could not create log"));

    let start = Instant::now();
    println!("Started parsing...");
    let (perm, graph, incoming_graph) = parse_graph("stgtregbz_ch.fmi")?;
    let duration = start.elapsed();
    println!("Loaded graph in {:.2?}", duration);

    let mut dijkstra = Dijkstra::new(&graph);
    let s = 377371;
    let t = 754742;
    print!("Dijkstra: ");
    let start = Instant::now();
    match dijkstra.shortest_path(perm[s as usize], perm[t as usize]) {
        Some(dist) => print!("Found a shortest path from {s} to {t}: {dist} "),
        None => print!("Did NOT find a path between {s} and {t} ")
    }
    let duration = start.elapsed();
    println!("[{:.2?}]", duration);

    let mut ch = CH::new(&graph, &incoming_graph);
    let s = 377371;
    let t = 754742;
    print!("CH: ");
    let start = Instant::now();
    match ch.shortest_path(perm[s as usize], perm[t as usize]) {
        Some(dist) => print!("Found a shortest path from {s} to {t}: {dist} "),
        None => print!("Did NOT find a path between {s} and {t} ")
    }
    let duration = start.elapsed();
    println!("[{:.2?}]", duration);

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::*;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_random_shortest_paths() {
        let (perm, graph, incoming_graph) = parse_graph("stgtregbz_ch.fmi").unwrap();
        let mut dijkstra = Dijkstra::new(&graph);
        let mut ch = CH::new(&graph, &incoming_graph);
        
        let mut rng = thread_rng();
        let mut s: u64 = rng.gen_range(0..graph.1.len() as u64 - 1);

        for i in 0..100 {
            let t: u64 = rng.gen_range(0..graph.1.len() as u64 - 1);
            
            // Dijkstra
            let start = Instant::now();
            let d = dijkstra.shortest_path(perm[s as usize], perm[t as usize]);
            let duration = start.elapsed();
            println!("Dijkstra [{:.2?}]", duration);

            // CH
            let start = Instant::now();
            let c = ch.shortest_path(perm[s as usize], perm[t as usize]);
            let duration = start.elapsed();
            println!("CH [{:.2?}]", duration);

            assert_eq!(d, c);
    
            // Only change s every 10th step
            if i % 10 == 0 {
                s = rng.gen_range(0..graph.1.len() as u64 - 1);
            }
        }
    }
}