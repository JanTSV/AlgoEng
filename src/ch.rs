use std::collections::BinaryHeap;
use std::process::exit;

use rayon::prelude::*;

use crate::graph::{OffsetArray, Edge};
use crate::dijkstra::{Distance, Dijkstra};

type Shortcut = (usize, usize, u64, usize, usize);

pub struct CH {
    graph: OffsetArray,

    // s -> t
    distances: Vec<Option<u64>>,
    heap: BinaryHeap<Distance>,
    visited: Vec<usize>,

    // t -> s
    incoming_distances: Vec<Option<u64>>,
    incoming_heap: BinaryHeap<Distance>
}

impl CH {
    pub fn new(graph: OffsetArray) -> Self {
        let n = graph.nodes.len();
        CH { graph, 
             distances: vec![None; n], 
             heap: BinaryHeap::new(), 
             visited: Vec::new(),
             incoming_distances: vec![None; n], 
             incoming_heap: BinaryHeap::new()
        }
    }

    pub fn get_graph(&self) -> &OffsetArray {
        &self.graph
    }

    pub fn set_graph(&mut self, graph: OffsetArray) {
        self.graph = graph
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
                    if self.incoming_distances[edge.from].is_none_or(|curr| weight + edge.weight < curr) && 
                       self.graph.nodes[edge.from].level > self.graph.nodes[id].level {
                        self.incoming_distances[edge.from] = Some(weight + edge.weight);
                        self.incoming_heap.push(Distance::new(weight + edge.weight, edge.from));
                        self.visited.push(edge.from);
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

    fn calc_shortcuts(graph: &OffsetArray, node: usize, contracted: &Vec<bool>) -> Vec<Shortcut> {
        // (from, to, weight, edge_id_a, edge_id_b)
        let mut shortcuts: Vec<Shortcut> = Vec::new();
        let mut dijkstra = Dijkstra::new(graph);

        for (edge_id_b, incoming_edge) in graph.incoming_edges(node).iter().enumerate() {
            let incoming_node = incoming_edge.from;
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
                    println!("this shouldn't happen: {} -> {}", incoming_node, outgoing_node);
                    exit(0);
                }
            }
        }

        shortcuts
    }

    fn contract_node(graph: &mut OffsetArray, node: usize, level: usize, contracted: &mut Vec<bool>) -> usize {
        let mut num_created = 0;

        for (from, to, weight, edge_id_a, edge_id_b) in Self::calc_shortcuts(graph, node, contracted) {
            
            graph.add_edge(Edge::new(from, to, weight, 0, -1, Some(edge_id_a), Some(edge_id_b)));
            num_created += 1;
        }

        graph.node_at_mut(node).level = level;
        contracted[node] = true;

        graph.build_offsets();
        num_created
    }

    fn contract_indep_set(graph: &mut OffsetArray, indep_set: &Vec<(Vec<Shortcut>, i64, usize)>, level: usize, threshold: i64, contracted: &mut Vec<bool>) -> (usize, usize) {
        let mut num_created = 0;
        let mut num_contracted = 0;


        // Create a new graph with the shortcuts
        for (shortcut, edge_diff, c) in indep_set {
            if *edge_diff <= threshold {
                // Add shortcuts
                for (from, to, weight, edge_id_a, edge_id_b) in shortcut {
                    graph.add_edge(Edge::new(*from, *to, *weight, 0, -1, Some(*edge_id_a), Some(*edge_id_b)));

                    num_created += 1;
                }

                //  Mark node as contracted and set level
                graph.node_at_mut(*c).level = level;
                contracted[*c] = true;
                num_contracted += 1;
            }
            
        }

        graph.build_offsets();

        (num_contracted, num_created)
    }

    pub fn batch_preprocess(&mut self) -> usize {
        let mut level = 0;
        let mut contracted = vec![false; self.graph.nodes.len()];
        let mut num_shortcuts = 0;
    
        while contracted.iter().any(|&c| !c) {
            // Find independent set
            //let mut indep_set = Self::find_independent_set(&self.graph, &contracted);
            let mut indep_set = Self::_find_independent_set(&self.graph, &contracted);
            if indep_set.is_empty() {
                break;
            }

            // Sort indep_set by edge difference
            //indep_set.sort_by_key(|x| x.1);
            indep_set.sort_by_key(|x| x.0);

            // Threshold is such that 3/4th of indep_set get contracted
            //let threshold = indep_set[3 * indep_set.len() / 4].1;
            let threshold = indep_set[3 * indep_set.len() / 4].0;

            //let (num_contracted, num_created, new_graph) = Self::contract_indep_set(&self.graph, &indep_set, level, threshold, &mut contracted);
            let (num_contracted, num_created) = Self::_contract_indep_set(&mut self.graph, &indep_set, level, threshold, &mut contracted);
            
            if num_contracted == 0 {
                break;
            }

            // Use new graph in next iteration
            println!("Created {} / Contracted {} at level {}", num_created, num_contracted, level);

            num_shortcuts += num_created;
            level += 1;
        }

        // Contract nodes if there are any left (a fallback jst in case, hopefully never)
        for i in 0..contracted.len() {
            if !contracted[i] {
                let num_created = Self::contract_node(&mut self.graph, i, level, &mut contracted);

                num_shortcuts += num_created;
                level += 1;

            }
        }

        println!("#created: {}", num_shortcuts);
        num_shortcuts
    }

    fn should_stall_forward(&self, node: usize) -> bool {
        let node_level = self.graph.nodes[node].level;
        let node_dist = match self.distances[node] {
            Some(d) => d,
            None => return false,
        };

        for edge in self.graph.incoming_edges(node) {
            let neighbor_level = self.graph.nodes[edge.from].level;

            // Only consider neighbors with higher level
            if neighbor_level > node_level {
                if let Some(alt_dist) = self.distances[edge.from] {
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
    
    fn find_independent_set(graph: &OffsetArray, contracted: &Vec<bool>) -> Vec<(Vec<Shortcut>, i64, usize)> {
        let mut independent_set: Vec<(Vec<Shortcut>, i64, usize)> = Vec::new();
        let mut blocked = vec![false; graph.nodes.len()];
    
        for node in 0..graph.nodes.len() {
            if contracted[node] || blocked[node] {
                continue;
            }

            let shortcuts = Self::calc_shortcuts(graph, node, contracted);
            let edge_diff = shortcuts.len() as i64 - Self::calc_edges_deleted(node, graph) as i64;
    
            independent_set.push((shortcuts, edge_diff, node));
            // Block its neighbors from being selected
            for edge in graph.outgoing_edges(node) {
                blocked[edge.to] = true;
            }
            for edge in graph.incoming_edges(node) {
                blocked[edge.from] = true;
            }
        }

        independent_set
    }

    fn calc_edges_deleted(node: usize, graph: &OffsetArray) -> usize {
        let incoming_edges = graph.incoming_edges(node).len();
        let outgoing_edges = graph.outgoing_edges(node).len();
        incoming_edges + outgoing_edges
    }


    fn _contract_indep_set(
        graph: &mut OffsetArray,
        indep_set: &Vec<(i64, usize)>,
        level: usize,
        threshold: i64,
        contracted: &mut Vec<bool>
    ) -> (usize, usize) {
    
        // Step 1: Compute shortcut additions in parallel
        let results: Vec<_> = indep_set
            .par_iter()
            .filter(|&&(edge_diff, _)| edge_diff <= threshold)
            .map(|&(_, node)| {
                let shortcuts = Self::calc_shortcuts(graph, node, contracted);
                (node, shortcuts)
            })
            .collect();
    
        let mut num_created = 0;
        let mut num_contracted = 0;
    
        // Step 2: Apply the modifications sequentially
        for (node, shortcuts) in results {
            for (from, to, weight, edge_id_a, edge_id_b) in shortcuts {
                graph.add_edge(Edge::new(from, to, weight, 0, -1, Some(edge_id_a), Some(edge_id_b)));
                num_created += 1;
            }
            
            graph.node_at_mut(node).level = level;
            contracted[node] = true;
            num_contracted += 1;
        }

        // Rebuild graph offsets
        graph.build_offsets();
    
        (num_contracted, num_created)
    }
    
    fn _find_independent_set(graph: &OffsetArray, contracted: &Vec<bool>) -> Vec<(i64, usize)> {
        let mut independent_set: Vec<(i64, usize)> = Vec::new();
        let mut blocked = vec![false; graph.nodes.len()];
    
        for node in 0..graph.nodes.len() {
            if contracted[node] || blocked[node] {
                continue;
            }

            independent_set.push((Self::_compute_edge_difference(node, graph, contracted), node));
            // Block its neighbors from being selected
            for edge in graph.outgoing_edges(node) {
                blocked[edge.to] = true;
            }
            for edge in graph.incoming_edges(node) {
                blocked[edge.from] = true;
            }
        }

        independent_set
    }

    fn _compute_edge_difference(node: usize, graph: &OffsetArray, contracted: &Vec<bool>) -> i64 {
        let mut shortcuts = 0;
    
        let incoming_edges = graph.incoming_edges(node);
        let outgoing_edges = graph.outgoing_edges(node);
    
        for in_edge in incoming_edges {
            let from = in_edge.from;
            if contracted[from] {
                continue;
            }
    
            for out_edge in outgoing_edges {
                let to = out_edge.to;
                if contracted[to] {
                    continue;
                }
    
                shortcuts += 1;
            }
        }
    
        // Number of edges deleted = all incident edges (incoming + outgoing)
        let edges_deleted = incoming_edges.len() + outgoing_edges.len();
    
        shortcuts as i64 - edges_deleted as i64
    }
}

#[cfg(test)]
mod test_ch {
    use std::time::Instant;

    use super::CH;
    use crate::dijkstra::Dijkstra;
    use crate::graph::OffsetArray;

    #[test]
    fn test_ch_without_stall_on_demand() {
        // Load graph with CH levels
        let start = Instant::now();
        println!("Started parsing...");
        let graph = OffsetArray::from_file("inputs/stgtregbz_ch.fmi").unwrap();
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
        let mut ch = CH::new(graph);
    
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
        let graph = OffsetArray::from_file("inputs/stgtregbz_ch.fmi").unwrap();
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
        let mut ch = CH::new(graph);
    
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
        // TODO
    }
}