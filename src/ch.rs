use std::collections::BinaryHeap;
use std::process::exit;

use crate::graph::OffsetArray;
use crate::dijkstra::{Distance, Dijkstra};

type Shortcut = (usize, usize, u64, usize, usize);

pub struct CH<'a> {
    graph: &'a OffsetArray,

    // s -> t
    distances: Vec<Option<u64>>,
    heap: BinaryHeap<Distance>,
    visited: Vec<usize>,

    // t -> s
    incoming_distances: Vec<Option<u64>>,
    incoming_heap: BinaryHeap<Distance>
}

impl<'a> CH<'a> {
    pub fn new(graph: &'a OffsetArray) -> Self {
        CH { graph, 
             distances: vec![None; graph.nodes.len()], 
             heap: BinaryHeap::new(), 
             visited: Vec::new(),
             incoming_distances: vec![None; graph.nodes.len()], 
             incoming_heap: BinaryHeap::new()
        }
    }

    pub fn shortest_path(&mut self, s: usize, t: usize, stall_on_demand: bool) -> Option<u64> {
        // Cleanup of previous run
        while let Some(node) = self.visited.pop() {
            self.distances[node] = None;
            self.incoming_distances[node] = None;
        }

        self.heap.clear();
        self.incoming_heap.clear();

        // Push s and t to heaps and set dists to 0
        self.heap.push(Distance::new(0, s));
        self.distances[s] = Some(0);
        self.visited.push(s);

        self.incoming_heap.push(Distance::new(0, t));
        self.incoming_distances[t] = Some(0);
        self.visited.push(t);

        while !self.heap.is_empty() || !self.incoming_heap.is_empty() {
            // Dijkstra from s
            if let Some(Distance { weight, id }) = self.heap.pop() {
                // Stall-on-demand
                if stall_on_demand && self.should_stall_forward(id) {
                    continue;
                }

                for edge in self.graph.outgoing_edges(id) {
                    // println!("CURRENT: {} {} EDGE: {} {}", id, self.graph.1[id].level, edge.to, self.graph.1[edge.to].level);
                    if self.distances[edge.to].is_none_or(|curr| weight + edge.weight < curr) && 
                       self.graph.nodes[edge.to].level > self.graph.nodes[id].level {
                        self.distances[edge.to] = Some(weight + edge.weight);
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

                for edge in self.graph.incoming_edges(id) {
                    if self.incoming_distances[edge.to].is_none_or(|curr| weight + edge.weight < curr) && 
                       self.graph.nodes[edge.to].level > self.graph.nodes[id].level {
                        self.incoming_distances[edge.to] = Some(weight + edge.weight);
                        self.incoming_heap.push(Distance::new(weight + edge.weight, edge.to));
                        self.visited.push(edge.to);
                    }
                }
            }
        }

        // Get minimal connecting node
        let mut distance: Option<u64> = None;
        for v in &self.visited {
            match (distance, self.distances[*v], self.incoming_distances[*v]) {
                (None, Some(d0), Some(d1)) => distance = Some(d0 + d1),
                (Some(d), Some(d0), Some(d1)) if d0 + d1 < d => distance = Some(d0 + d1),
                _ => continue
            }
        }

        distance
    }

    pub fn batch_preprocess(graph: &mut OffsetArray) -> usize {
        let mut dijkstra = Dijkstra::unsafe_new(graph);
        let mut level = 0;
        let mut contracted = vec![false; graph.nodes.len()];
        let mut overall_contracted = 0;
        const THRESHOLD: i64 = 1;
    
        while contracted.iter().any(|&c| !c) {
            // println!("{} / {}", _i, contracted.len());
            let indep_set = Self::find_independent_set(graph, &contracted);
            if indep_set.is_empty() {
                break;
            }
    
            let num_contracted = Self::contract_independent_set(graph, &mut contracted, &mut dijkstra, indep_set, level, THRESHOLD);
            if num_contracted == 0 {
                break;
            }

            overall_contracted += num_contracted;
            level += 1;
        }

        // If there are still uncontracted nodes
        // then there cannot be a new independet set.
        // Since they are not independent, contract them one by one
        let remaining_nodes: Vec<usize> = (0..graph.nodes.len())
            .filter(|&node| !contracted[node])
            .map(|node| node)
            .collect();

        for node in remaining_nodes {
            overall_contracted += Self::contract_node(node, graph, &mut contracted, &mut dijkstra, level, THRESHOLD);
            level += 1;
        }

        println!("#contracted: {}", overall_contracted);
        overall_contracted
    }

    fn should_stall_forward(&self, node: usize) -> bool {
        let node_level = self.graph.nodes[node].level;
        let node_dist = match self.distances[node] {
            Some(d) => d,
            None => return false,
        };

        for edge in self.graph.incoming_edges(node) {
            let neighbor_level = self.graph.nodes[edge.to].level;

            // Only consider neighbors with higher level
            if neighbor_level > node_level {
                if let Some(alt_dist) = self.distances[edge.to] {
                    if alt_dist + edge.weight < node_dist {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn should_stall_backward(&self, node: usize) -> bool {
        let node_level = self.graph.nodes[node].level;
        let node_dist = match self.incoming_distances[node] {
            Some(d) => d,
            None => return false,
        };

        for edge in self.graph.outgoing_edges(node) {
            let neighbor_level = self.graph.nodes[edge.to].level;

            if neighbor_level > node_level {
                if let Some(alt_dist) = self.incoming_distances[edge.to] {
                    if alt_dist + edge.weight < node_dist {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn find_independent_set(graph: &OffsetArray, contracted: &Vec<bool>) -> Vec<usize> {
        let mut independent_set = Vec::new();
        let mut blocked = vec![false; graph.nodes.len()];
    
        for node in 0..graph.nodes.len() {
            if contracted[node] || blocked[node] {
                continue;
            }
    
            independent_set.push(node);
            // Block its neighbors from being selected
            for edge in graph.outgoing_edges(node) {
                blocked[edge.to] = true;
            }
            for edge in graph.incoming_edges(node) {
                blocked[edge.to] = true;
            }
        }
        independent_set
    }

    fn contract_independent_set(
        graph: &mut OffsetArray,
        contracted: &mut Vec<bool>,
        dijkstra: &mut Dijkstra,
        indep_set: Vec<usize>,
        level: usize,
        threshold: i64,
    ) -> usize {
        let mut num_contracted = 0;
    
        for &node in &indep_set {
            num_contracted += Self::contract_node(node, graph, contracted, dijkstra, level, threshold);
        }
    
        num_contracted
    }

    fn contract_node(node: usize,
        graph: &mut OffsetArray,
        contracted: &mut Vec<bool>,
        dijkstra: &mut Dijkstra,
        level: usize,
        threshold: i64) -> usize {
            let mut num_contracted = 0;
            let diff = Self::compute_edge_difference(node, graph);
            if diff <= threshold {
                let shortcuts = Self::calc_shortcuts(node, graph, contracted, dijkstra);
                for (from, to, weight, edge_id_a, edge_id_b) in shortcuts {
                    // TODO: add shortcuts and patch graph
                    // graph.edges[from as usize].push(Edge::new(to, weight, Some(edge_id_a), Some(edge_id_b)));
                    // graph.adj_edges[to as usize].push(Edge::new(from, weight, Some(edge_id_a), Some(edge_id_b)));
                }
                graph.node_at_mut(node).level = level;
                contracted[node] = true;
                num_contracted += 1;
            }
            num_contracted
    }

    fn calc_shortcuts(node: usize, graph: &mut OffsetArray, contracted: &Vec<bool>, dijkstra: &mut Dijkstra) -> Vec<Shortcut> {
        // (from, to, weight, edge_id_a, edge_id_b)
        let mut shortcuts : Vec<Shortcut> = Vec::new();

        for (edge_id_b, incoming_edge) in graph.incoming_edges(node).iter().enumerate() {
            let incoming_node = incoming_edge.to;
            for (edge_id_a, outgoing_edge) in graph.outgoing_edges(node).iter().enumerate() {
                let outgoing_node = outgoing_edge.to;

                if contracted[incoming_node] || contracted[outgoing_node] {
                    continue;
                }

                if let Some(shortest_path) = dijkstra.shortest_path_consider_contraction(incoming_node,outgoing_node, contracted) {
                    let direct_distance = incoming_edge.weight + outgoing_edge.weight;
                    if shortest_path >= direct_distance {
                        shortcuts.push((incoming_node, outgoing_node, direct_distance, edge_id_a, edge_id_b));
                    }
                } else {
                    println!("WHOOP: {} -> {}", incoming_node, outgoing_node);
                    exit(0);
                }
            }
        }

        shortcuts
    }

    fn calc_edges_deleted(node: usize, graph: &OffsetArray) -> usize {
        let incoming_edges = graph.outgoing_edges(node).len();
        let outgoing_edges = graph.incoming_edges(node).len();
        incoming_edges + outgoing_edges
    }

    fn max_shortcuts_created(node: usize, graph: &OffsetArray) -> usize {
        let incoming_edges = graph.outgoing_edges(node).len();
        let outgoing_edges = graph.incoming_edges(node).len();
        incoming_edges * outgoing_edges
    }

    fn compute_edge_difference(node: usize, graph: &OffsetArray) -> i64 {
        let max_shortcuts_created = Self::max_shortcuts_created(node, graph) as i64;
        let edges_deleted = Self::calc_edges_deleted(node, graph) as i64;
        max_shortcuts_created - edges_deleted
    }
}

#[cfg(test)]
mod test_ch {
    use std::time::Instant;

    use super::CH;
    use crate::perm::Permutation;
    use crate::reader::parse_graph;
    use crate::dijkstra::Dijkstra;

    #[test]
    fn test_ch_without_stall_on_demand() {
        // Load graph with CH levels
        let start = Instant::now();
        println!("Started parsing...");
        let graph = parse_graph("inputs/stgtregbz_ch.fmi").unwrap();
        let duration = start.elapsed();
        println!("Loaded graph in {:.2?}", duration);

        // Test Dijkstra vs CH
        let mut dijkstra = Dijkstra::new(&graph);
        const START: usize = 377371;
        const TARGET: usize = 754742;
        print!("Dijkstra: ");
        let start = Instant::now();
        let dijkstra_found = dijkstra.shortest_path(START, TARGET);
        match dijkstra_found {
            Some(dist) => print!("Found a shortest path from {START} to {TARGET}: {dist} "),
            None => print!("Did NOT find a path between {START} and {TARGET} ")
        }
        let duration = start.elapsed();
        println!("[{:.2?}]", duration);
        let mut ch = CH::new(&graph);
    
        print!("CH (without stall-on-demand): ");
        let start = Instant::now();
        let ch_found = ch.shortest_path(START, TARGET, false);
        match ch_found {
            Some(dist) => print!("Found a shortest path from {START} to {TARGET}: {dist} "),
            None => print!("Did NOT find a path between {START} and {TARGET} ")
        }
        let duration = start.elapsed();
        println!("[{:.2?}]", duration);

        assert_eq!(dijkstra_found, ch_found);
    }

    
    #[test]
    fn test_ch_with_stall_on_demand() {
        // Load graph with CH levels
        let start = Instant::now();
        println!("Started parsing...");
        let graph = parse_graph("inputs/stgtregbz_ch.fmi").unwrap();
        let duration = start.elapsed();
        println!("Loaded graph in {:.2?}", duration);

        // Test Dijkstra vs CH
        let mut dijkstra = Dijkstra::new(&graph);
        const START: usize = 377371;
        const TARGET: usize = 754742;
        print!("Dijkstra: ");
        let start = Instant::now();
        let dijkstra_found = dijkstra.shortest_path(START, TARGET);
        match dijkstra_found {
            Some(dist) => print!("Found a shortest path from {START} to {TARGET}: {dist} "),
            None => print!("Did NOT find a path between {START} and {TARGET} ")
        }
        let duration = start.elapsed();
        println!("[{:.2?}]", duration);
        let mut ch = CH::new(&graph);
    
        print!("CH (with stall-on-demand): ");
        let start = Instant::now();
        let ch_found = ch.shortest_path(START, TARGET, true);
        match ch_found {
            Some(dist) => print!("Found a shortest path from {START} to {TARGET}: {dist} "),
            None => print!("Did NOT find a path between {START} and {TARGET} ")
        }
        let duration = start.elapsed();
        println!("[{:.2?}]", duration);

        assert_eq!(dijkstra_found, ch_found);
    }

    
    #[test]
    fn test_ch_with_stall_on_demand_and_sorted_by_level() {
        // Load graph with CH levels
        let start = Instant::now();
        println!("Started parsing...");
        let graph = parse_graph("inputs/stgtregbz_ch.fmi").unwrap();
        let duration = start.elapsed();
        println!("Loaded graph in {:.2?}", duration);

        // Permutate graph
        let perm = Permutation::by_level(&graph);
        let graph = perm.permutate_graph(&graph);

        // Test Dijkstra vs CH
        let mut dijkstra = Dijkstra::new(&graph);
        const START: usize = 377371;
        const TARGET: usize = 754742;
        print!("Dijkstra: ");
        let start = Instant::now();
        let dijkstra_found = dijkstra.shortest_path(perm.from(START), perm.from(TARGET));
        match dijkstra_found {
            Some(dist) => print!("Found a shortest path from {START} to {TARGET}: {dist} "),
            None => print!("Did NOT find a path between {START} and {TARGET} ")
        }
        let duration = start.elapsed();
        println!("[{:.2?}]", duration);
        let mut ch = CH::new(&graph);
    
        print!("CH (with stall-on-demand): ");
        let start = Instant::now();
        let ch_found = ch.shortest_path(perm.from(START), perm.from(TARGET), true);
        match ch_found {
            Some(dist) => print!("Found a shortest path from {START} to {TARGET}: {dist} "),
            None => print!("Did NOT find a path between {START} and {TARGET} ")
        }
        let duration = start.elapsed();
        println!("[{:.2?}]", duration);

        assert_eq!(dijkstra_found, ch_found);
    }
}