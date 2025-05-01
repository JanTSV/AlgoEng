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
    id: u64,
    level: Option<u64>
}

impl Node {
    pub fn new(id: u64, level: Option<u64>) -> Self {
        Node { id, level }
    }
}

struct Graph {
    edges: Vec<Vec<Edge>>,
    adj_edges: Vec<Vec<Edge>>,
    nodes: Vec<Node>
}

impl Graph {
    pub fn new(edges: Vec<Vec<Edge>>, adj_edges: Vec<Vec<Edge>>, nodes: Vec<Node>) -> Self {
        assert_eq!(edges.len(), adj_edges.len());
        assert_eq!(nodes.len(), edges.len());
        Graph { edges, adj_edges, nodes }
    }
}

type Level = (u64, Option<u64>);

fn parse_graph(filename: &str) -> Result<(Vec<u64>, Graph), Box<dyn Error>> {
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
    let mut nodes: Vec<Node> = vec![];
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

        nodes.push(Node::new(id, level));
    }
    
    // Sort nodes by level
    nodes.sort_by_key(|node: &Node| node.level);
    
    // Map original node IDs to new indices after sorting
    let mut id_to_new_index = vec![0u64; nodes.len()];
    for (new_index, Node {id: old_id, level: _}) in nodes.iter().enumerate() {
        id_to_new_index[*old_id as usize] = new_index as u64;
    }


    // Build adjacency lists
    let mut edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes as usize];
    let mut adj_edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes as usize];

    for _ in 0..num_edges {
        let line = lines.next().ok_or("Missing edge line")?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 3 {
            return Err(format!("Malformed edge line {}, parts: {}", line, parts.len()).into());
        }

        let source: u64 = parts[0].parse()?;
        let target: u64 = parts[1].parse()?;
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

        // Add to correct sorted node
        let source = id_to_new_index[source as usize];
        let target = id_to_new_index[target as usize];

        edges[source as usize].push(Edge::new(target, weight, edge_id_a, edge_id_b));
        adj_edges[target as usize].push(Edge::new(source, weight, edge_id_b, edge_id_a));
    }

    Ok((id_to_new_index, Graph::new(edges, adj_edges, nodes)))
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

type Shortcut = (u64, u64, u64, u64, u64);

struct CH<'a> {
    graph: &'a Graph,

    // s -> t
    distances: Vec<Option<u64>>,
    heap: BinaryHeap<Distance>,
    visited: Vec<u64>,

    // t -> s
    incoming_distances: Vec<Option<u64>>,
    incoming_heap: BinaryHeap<Distance>
}

impl<'a> CH<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        CH { graph, 
             distances: vec![None; graph.nodes.len()], 
             heap: BinaryHeap::new(), 
             visited: Vec::new(),
             incoming_distances: vec![None; graph.nodes.len()], 
             incoming_heap: BinaryHeap::new()
        }
    }

    fn should_stall_forward(&self, node: u64) -> bool {
        let node_level = self.graph.nodes[node as usize].level;
        let node_dist = match self.distances[node as usize] {
            Some(d) => d,
            None => return false,
        };

        for edge in &self.graph.adj_edges[node as usize] {
            let neighbor_level = self.graph.nodes[edge.to as usize].level;

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
        let node_level = self.graph.nodes[node as usize].level;
        let node_dist = match self.incoming_distances[node as usize] {
            Some(d) => d,
            None => return false,
        };

        for edge in &self.graph.edges[node as usize] {
            let neighbor_level = self.graph.nodes[edge.to as usize].level;

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

                for edge in &self.graph.edges[id as usize] {
                    // println!("CURRENT: {} {} EDGE: {} {}", id, self.graph.1[id as usize].level, edge.to, self.graph.1[edge.to as usize].level);
                    if self.distances[edge.to as usize].is_none_or(|curr| weight + edge.weight < curr) && 
                       self.graph.nodes[edge.to as usize].level >= self.graph.nodes[id as usize].level {
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

                for edge in &self.graph.adj_edges[id as usize] {
                    if self.incoming_distances[edge.to as usize].is_none_or(|curr| weight + edge.weight < curr) && 
                       self.graph.nodes[edge.to as usize].level >= self.graph.nodes[id as usize].level {
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

    pub fn preprocess(graph: &mut Graph, perm: &Vec<u64>) {
        let l = graph.nodes.len();
        let mut visited = vec![false; l];
        let mut contracted = vec![false; l];
        let mut number_added_shortcuts = 0;
        let mut dijkstra = Dijkstra::unsafe_new(graph as *const Graph); // Create Dijkstra instance here

        for node in 0..l {
            println!("Preprocessing at {} out of {}: [{}%]", node, l, (node * 100) / l);

            if !visited[node] {
                let mut level = 0;
                let mut subgraph = Self::dfs(graph, node as u64, &mut visited);

                // Sort subgraph by increasing #max shortcuts created - #edges deleted
                subgraph.sort_by_key(|node| {
                    let max_shortcuts_created = Self::max_shortcuts_created(*node, graph);
                    let edges_deleted = Self::calc_edges_deleted(*node, graph);
                    max_shortcuts_created as i64 - edges_deleted as i64
                });

                // Iterate over sorted subgraph and contract nodes
                for (_i, node) in subgraph.iter().enumerate() {
                    println!("  Subgraph: [{} / {}]", _i, subgraph.len());
                    let shortcuts = Self::calc_shortcuts(*node, graph, &contracted, &mut dijkstra); // Pass the Dijkstra instance

                    for (from, to, weight, edge_id_a, edge_id_b) in shortcuts {
                        // Add shortcut to graphs
                        graph.edges[from as usize].push(Edge::new(to, weight, Some(edge_id_a), Some(edge_id_b)));
                        graph.adj_edges[to as usize].push(Edge::new(from, weight, Some(edge_id_a), Some(edge_id_b)));
                        number_added_shortcuts += 1;
                    }

                    // Set level of node
                    graph.nodes[*node as usize].level = Some(level);
                    level += 1;

                    // Mark node as contracted
                    contracted[*node as usize] = true;
                }
            }
        }
    }

    fn calc_shortcuts(node: u64, graph: &Graph, contracted: &Vec<bool>, dijkstra: &mut Dijkstra) -> Vec<Shortcut> {
        // (from, to, weight, edge_id_a, edge_id_b)
        let mut shortcuts : Vec<Shortcut> = Vec::new();

        for (edge_id_a, incoming_edge) in graph.adj_edges[node as usize].iter().enumerate() {
            let incoming_node = incoming_edge.to;
            for (edge_id_b, outgoing_edge) in graph.edges[node as usize].iter().enumerate() {
                let outgoing_node = outgoing_edge.to;
                // Skip already contracted nodes
                if contracted[incoming_node as usize] || contracted[outgoing_node as usize] {
                    continue;
                }

                if let Some(shortest_path) = dijkstra.shortest_path(incoming_node, outgoing_node, Some(contracted)) {
                    let direct_distance = incoming_edge.weight + outgoing_edge.weight;
                    if shortest_path >= direct_distance {
                        shortcuts.push((incoming_node, outgoing_node, direct_distance, edge_id_a as u64, edge_id_b as u64));
                    }
                }
            }
        }

        shortcuts
    }

    fn calc_edges_deleted(node: u64, graph: &Graph) -> u64 {
        let incoming_edges = graph.adj_edges[node as usize].len() as u64;
        let outgoing_edges = graph.edges[node as usize].len() as u64;
        incoming_edges + outgoing_edges
    }

    fn max_shortcuts_created(node: u64, graph: &Graph) -> u64 {
        let incoming_edges = graph.adj_edges[node as usize].len() as u64;
        let outgoing_edges = graph.edges[node as usize].len() as u64;
        incoming_edges * outgoing_edges
    }

    fn dfs(graph: &Graph, start: u64, visited: &mut [bool]) -> Vec<u64> {
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
            for edge in &graph.edges[node] {
                let neighbor = edge.to;
                if !visited[neighbor as usize] {
                    stack.push(neighbor);
                }
            }
    
            for edge in &graph.adj_edges[node] {
                let neighbor = edge.to;
                if !visited[neighbor as usize] {
                    stack.push(neighbor);
                }
            }
        }

        subgraph
    }
}

struct Dijkstra<'a> {
    graph: &'a Graph,
    distances: Vec<Option<u64>>,
    heap: BinaryHeap<Distance>,
    visited: Vec<u64>
}

impl<'a> Dijkstra<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        Dijkstra { graph, distances: vec![None; graph.nodes.len()], heap: BinaryHeap::new(), visited: Vec::new() }
    }

    pub fn unsafe_new(graph_ptr: *const Graph) -> Self{
        unsafe {
            let graph = &*graph_ptr;
            Self::new(graph)
        }
    }

    pub fn shortest_path(&mut self, s: u64, t: u64, contracted: Option<&Vec<bool>>) -> Option<u64> {
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

            if let Some(contracted) = contracted {
                if contracted[id as usize] {
                    continue;
                }
            }

            for edge in &self.graph.edges[id as usize] {
                if let Some(contracted) = contracted {
                    if contracted[edge.to as usize] {
                        continue;
                    }
                }
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
    if true {
        // Load graph with CH levels
        let mut log = BufWriter::new(File::create("dump.txt").expect("Could not create log"));

        let start = Instant::now();
        println!("Started parsing...");
        let (perm, graph) = parse_graph("stgtregbz_ch.fmi")?;
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
        match dijkstra.shortest_path(perm[s as usize], perm[t as usize], None) {
            Some(dist) => print!("Found a shortest path from {s} to {t}: {dist} "),
            None => print!("Did NOT find a path between {s} and {t} ")
        }
        let duration = start.elapsed();
        println!("[{:.2?}]", duration);

        let mut ch = CH::new(&graph);
    
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
        exit(0);
    }
    // Task 2: Run own CH preprocessing
    
    // TODO: IDK remove this or hmm
    let start = Instant::now();
    println!("Started parsing...");
    let (prep_perm, mut prep_graph) = parse_graph("inputs/MV.fmi")?;
    let duration = start.elapsed();
    println!("Loaded graph in {:.2?}", duration);

    let start = Instant::now();
    println!("Starting CH preprocessing...");
    CH::preprocess(&mut prep_graph, &prep_perm);
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
        let (perm, graph) = parse_graph("stgtregbz_ch.fmi").unwrap();
        let mut dijkstra = Dijkstra::new(&graph);
        let mut ch = CH::new(&graph);
        
        let mut rng = thread_rng();
        let mut s: u64 = rng.gen_range(0..graph.nodes.len() as u64);

        for i in 0..100 {
            let t: u64 = rng.gen_range(0..graph.nodes.len() as u64);
            
            // Dijkstra
            let start = Instant::now();
            let d = dijkstra.shortest_path(perm[s as usize], perm[t as usize], None);
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
                s = rng.gen_range(0..graph.nodes.len() as u64);
            }
        }
    }
}