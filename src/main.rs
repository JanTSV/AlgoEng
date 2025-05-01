use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::error::Error;
use std::process::exit;
use std::time::Instant;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
struct Node {
    offset: u64,
    level: Option<u64>
}

impl Node {
    pub fn new(offset: u64, level: Option<u64>) -> Self {
        Node { offset, level }
    }
}

type OffsetArray = (Vec<Edge>, Vec<Node>);
type Level = (u64, Option<u64>);

fn create_offset_array(adj_list: Vec<Vec<Edge>>, levels: &Vec<Level>) -> OffsetArray {
    let mut flat_edges: Vec<Edge> = Vec::new();
    let mut nodes: Vec<Node> = Vec::with_capacity(adj_list.len() + 1);
    let mut current_offset = 0u64;

    nodes.push(Node::new(current_offset, levels[0].1));
    assert_eq!(adj_list.len(), levels.len());
    for (i, edges) in adj_list.iter().enumerate() {
        current_offset += edges.len() as u64;
        flat_edges.extend(edges.clone());
        nodes.push(Node::new(current_offset, levels.get(i + 1).map(|lvl| lvl.1).unwrap_or(Some(0))));
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
        
        if parts.len() < 5 {
            return Err(format!("Malformed node line {}, parts: {}", line, parts.len()).into());
        }

        let id: u64 = parts[0].parse()?;
        assert_eq!(id, _i);
        let level: Option<u64> = if parts.len() >= 6 {
            parts[5].parse::<u64>().ok()
        } else {
            None
        };

        levels.push((id, level));
    }
    

    // Build adjacency lists
    let mut outgoing_edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes as usize];
    let mut incoming_edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes as usize];

    for _ in 0..num_edges {
        let line = lines.next().ok_or("Missing edge line")?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 3 {
            return Err(format!("Malformed edge line {}, parts: {}", line, parts.len()).into());
        }

        let source: u64 = levels[parts[0].parse::<usize>()?].0;
        let target: u64 = levels[parts[1].parse::<usize>()?].0;
        let weight: u64 = parts[2].parse()?;
        let edge_id_a: Option<u64> = if parts.len() > 5 {
            parts[5].parse().ok()
        } else {
            None
        };
        let edge_id_b: Option<u64> = if parts.len() > 6 {
            parts[6].parse().ok()
        } else {
            None
        };

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

    fn should_stall_forward(&self, node: u64) -> bool {
        let node_level = self.graph.1[node as usize].level;
        let node_dist = match self.distances[node as usize] {
            Some(d) => d,
            None => return false,
        };

        for i in self.incoming_graph.1[node as usize].offset..self.incoming_graph.1[node as usize + 1].offset {
            let edge = &self.incoming_graph.0[i as usize];
            let neighbor_level = self.graph.1[edge.to as usize].level;

            // Only consider neighbors with higher level
            if neighbor_level > node_level {
                if let Some(alt_dist) = self.distances[edge.to as usize] {
                    if alt_dist + edge.weight < node_dist {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn should_stall_backward(&self, node: u64) -> bool {
        let node_level = self.incoming_graph.1[node as usize].level;
        let node_dist = match self.incoming_distances[node as usize] {
            Some(d) => d,
            None => return false,
        };

        for i in self.graph.1[node as usize].offset..self.graph.1[node as usize + 1].offset {
            let edge = &self.graph.0[i as usize];
            let neighbor_level = self.incoming_graph.1[edge.to as usize].level;

            if neighbor_level > node_level {
                if let Some(alt_dist) = self.incoming_distances[edge.to as usize] {
                    if alt_dist + edge.weight < node_dist {
                        return true;
                    }
                }
            }
        }
        false
    }

    pub fn shortest_path(&mut self, s: u64, t: u64, stall_on_demand: bool) -> Option<u64> {
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
                // Stall-on-demand
                if stall_on_demand && self.should_stall_forward(id) {
                    continue;
                }

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
                // Stall-on-demand
                if stall_on_demand && self.should_stall_backward(id) {
                    continue;
                }

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

    pub fn preprocess(graph: &mut OffsetArray, incoming_graph: &mut OffsetArray, perm: &Vec<u64>) {
        // TODO: Preprocessing needs to be called on every weak component
        // Use dfs to reach weak components and call sub function on this comp
        // dfs() needs to return list of nodes in component (Independent set)
        // Sort independent set by edge-difference (#shortcuts created - #edges deleted)
        // Contract nodes (u) in that order with Dijkstras (shortcuts only if Dijkstra(neighbor_i, neighbor_j)
        //  > c(u, neighbor_i) + c(u, neighbor_j). i and j are edgeA and edgeB
        let mut visited = vec![false; graph.1.len() - 1];

        for node in 0..visited.len() {
            if !visited[node] {
                let mut subgraph = Self::dfs(graph, incoming_graph, node as u64, &mut visited);

                // Sort subgraph by increasing #shortcuts created - #edges deleted
                subgraph.sort_by_key(|node| {
                    let edges_deleted = Self::calc_edges_deleted(*node, graph, incoming_graph);
                    let shortcuts_created = Self::calc_shortcuts(*node, graph, incoming_graph).len();
                    
                    shortcuts_created as i64 - edges_deleted as i64
                });

                // TODO: Contract nodes
                for node in subgraph {

                }
            }
        }

    }

    fn calc_shortcuts(node: u64, graph: &OffsetArray, incoming_graph: &OffsetArray) -> Vec<(u64, u64, u64, u64, u64)> {
        let mut dijkstra = Dijkstra::new(graph);
        // (from, to, weight, edge_id_a, edge_id_b)
        let mut shortcuts : Vec<(u64, u64, u64, u64, u64)> = Vec::new();

        for incoming_edge in incoming_graph.1[node as usize].offset..incoming_graph.1[node as usize + 1].offset {
            let incoming_node = incoming_graph.0[incoming_edge as usize].to;
            for outgoing_edge in graph.1[node as usize].offset..graph.1[node as usize + 1].offset {
                let outgoing_node = graph.0[outgoing_edge as usize].to;
                if let Some(shortest_path) = dijkstra.shortest_path(incoming_node, outgoing_node) {
                    let direct_distance = incoming_graph.0[incoming_edge as usize].weight + graph.0[outgoing_edge as usize].weight;
                    if shortest_path >= direct_distance {
                        shortcuts.push((incoming_node, outgoing_node, direct_distance, incoming_edge, outgoing_edge));
                    }
                }
            }
        }

        shortcuts
    }

    fn calc_edges_deleted(node: u64, graph: &OffsetArray, incoming_graph: &OffsetArray) -> u64 {
        let incoming_edges = incoming_graph.1[node as usize + 1].offset - incoming_graph.1[node as usize].offset;
        let outgoing_edges = graph.1[node as usize + 1].offset - graph.1[node as usize].offset;
        incoming_edges + outgoing_edges
    }

    fn dfs(graph: &OffsetArray, incoming_graph: &OffsetArray, start: u64, visited: &mut [bool]) -> Vec<u64> {
        let mut stack = Vec::new();
        let mut subgraph = Vec::new();
        stack.push(start);
    
        while let Some(node) = stack.pop() {
            subgraph.push(node);
            let node = node as usize;
            if visited[node] {
                continue;
            }
    
            visited[node] = true;
    
            // Traverse bidirectionally because we need weak comps
            for i in graph.1[node].offset..graph.1[node + 1].offset {
                let neighbor = graph.0[i as usize].to;
                if !visited[neighbor as usize] {
                    stack.push(neighbor);
                }
            }
    
            for i in incoming_graph.1[node].offset..incoming_graph.1[node + 1].offset {
                let neighbor = incoming_graph.0[i as usize].to;
                if !visited[neighbor as usize] {
                    stack.push(neighbor);
                }
            }
        }

        subgraph
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
    // Task 1
    if false {
        // Load graph with CH levels
        let mut log = BufWriter::new(File::create("dump.txt").expect("Could not create log"));

        let start = Instant::now();
        println!("Started parsing...");
        let (perm, graph, incoming_graph) = parse_graph("stgtregbz_ch.fmi")?;
        let duration = start.elapsed();
        println!("Loaded graph in {:.2?}", duration);

        // Test Dijkstra vs CH
        let mut dijkstra = Dijkstra::new(&graph);
        // let s = 214733;
        // let t = 429466;
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
    
        print!("CH (without stall-on-demand): ");
        let start = Instant::now();
        match ch.shortest_path(perm[s as usize], perm[t as usize], false) {
            Some(dist) => print!("Found a shortest path from {s} to {t}: {dist} "),
            None => print!("Did NOT find a path between {s} and {t} ")
        }
        let duration = start.elapsed();
        println!("[{:.2?}]", duration);

        print!("CH (with stall-on-demand): ");
        let start = Instant::now();
        match ch.shortest_path(perm[s as usize], perm[t as usize], true) {
            Some(dist) => print!("Found a shortest path from {s} to {t}: {dist} "),
            None => print!("Did NOT find a path between {s} and {t} ")
        }
        let duration = start.elapsed();
        println!("[{:.2?}]", duration);
    }
    // Task 2: Run own CH preprocessing
    
    // TODO: IDK remove this or hmm
    let start = Instant::now();
    println!("Started parsing...");
    let (prep_perm, mut prep_graph, mut prep_incoming_graph) = parse_graph("inputs/MV.fmi")?;
    let duration = start.elapsed();
    println!("Loaded graph in {:.2?}", duration);

    let start = Instant::now();
    println!("Starting CH preprocessing...");
    CH::preprocess(&mut prep_graph, &mut prep_incoming_graph, &prep_perm);
    let duration = start.elapsed();
    println!("Preprocessed in {:.2?}", duration);


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
            let c = ch.shortest_path(perm[s as usize], perm[t as usize], true);
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