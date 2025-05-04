use std::collections::BinaryHeap;
use std::process::exit;

use rayon::prelude::*;

use crate::graph::{Graph};
use crate::dijkstra::{Distance, Dijkstra};

type Shortcut = (usize, usize, u64, usize, usize);

pub struct CH {
    graph: Graph,

    // s -> t
    distances: Vec<Option<u64>>,
    heap: BinaryHeap<Distance>,
    visited: Vec<usize>,

    // t -> s
    incoming_distances: Vec<Option<u64>>,
    incoming_heap: BinaryHeap<Distance>
}

impl CH {
    pub fn new(graph: Graph) -> Self {
        let n = graph.num_nodes();
        CH { graph, 
             distances: vec![None; n], 
             heap: BinaryHeap::new(), 
             visited: Vec::new(),
             incoming_distances: vec![None; n], 
             incoming_heap: BinaryHeap::new()
        }
    }

    pub fn get_graph(&self) -> &Graph {
        &self.graph
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
                    if self.distances[edge.0].is_none_or(|curr| weight + edge.1 < curr) && 
                       self.graph.node_at(edge.0).level > self.graph.node_at(id).level {
                        self.distances[edge.0] = Some(weight + edge.1);
                        self.heap.push(Distance::new(weight + edge.1, edge.0));
                        self.visited.push(edge.0);
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
                    if self.incoming_distances[edge.0].is_none_or(|curr: u64| weight + edge.1 < curr) && 
                       self.graph.node_at(edge.0).level > self.graph.node_at(id).level {
                        self.incoming_distances[edge.0] = Some(weight + edge.1);
                        self.incoming_heap.push(Distance::new(weight + edge.1, edge.0));
                        self.visited.push(edge.0);
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

    fn calc_shortcuts(&self, dijkstra: &mut Dijkstra, node: usize, contracted: &[u64]) -> Vec<Shortcut> {
        // (from, to, weight, edge_id_a, edge_id_b)
        let mut shortcuts: Vec<Shortcut> = Vec::new();

        let predecessors = self
            .graph
            .incoming_edges(node)
            .filter(|(id, _)| contracted[*id/ 64] & (1 << (id % 64)) == 0)
            .collect::<Vec<_>>();

        let successors = self
            .graph
            .outgoing_edges(node)
            .filter(|(id, _)| contracted[*id / 64] & (1 << (id % 64)) == 0)
            .collect::<Vec<_>>();

        let successor_ids: Vec<_> = successors
            .iter()
            .map(|(id, _)| *id)
            .collect();

        if predecessors.is_empty() || successors.is_empty() {
            return shortcuts;
        }

        for (edge_id_b, (pred_id, pred_weight)) in predecessors.iter().enumerate() {
            let shortest_distances = dijkstra.shortest_path_consider_contraction(*pred_id, &successor_ids, contracted);
            assert_eq!(shortest_distances.len(), successors.len());

            for (edge_id_a, (succ_id, succ_weight)) in successors.iter().enumerate() {
                let direct_distance = *pred_weight + *succ_weight;
                if let Some((_, shortest_distance)) = shortest_distances.iter().find(|(id, _)| *succ_id == *id) {
                    if *shortest_distance >= direct_distance {
                        shortcuts.push((*pred_id, *succ_id, direct_distance, edge_id_a, edge_id_b));
                    }
                }
            }
        }

        shortcuts
    }

    fn contract_indep_set(
        &mut self,
        indep_set: &Vec<(isize, usize)>,
        level: usize,
        contracted: &mut [u64],
        nodes: &mut Vec<usize>
    ) -> (usize, usize) {
    
        // Compute shortcuts for part of independent set with low edge diff
        let n = indep_set.len().div_ceil(6);
        let sub_indep_set = &indep_set[..n];
        let chunk_size = n;// .div_ceil(rayon::current_num_threads());
    
        let results: Vec<(usize, Vec<Shortcut>)> = sub_indep_set
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                let mut dijkstra = Dijkstra::new(&self.graph);
                chunk.iter().map(|&(_, node)| {
                    let shortcuts = self.calc_shortcuts(&mut dijkstra, node, contracted);
                    (node, shortcuts)
                }).collect::<Vec<_>>() // collect per chunk to allow flat_map
            })
            .collect();
    
        let mut num_created = 0;
        let num_contracted = results.len();
    
        // Add shortcuts sequentially (can parallelize too if thread-safe)
        for (node, shortcuts) in results {
            for (from, to, weight, _edge_id_a, _edge_id_b) in shortcuts {
                self.graph.add_edge(from, to, weight);
                num_created += 1;
            }
    
            self.graph.node_at_mut(node).level = level;
            contracted[node / 64] |= 1 << (node % 64);
        }
    
        for i in n..indep_set.len() {
            let node = indep_set[i].1;
            assert!((contracted[node / 64]) & (1 << (node % 64)) == 0);
            nodes.push(node);
        }
    
        (num_contracted, num_created)
    }

    pub fn batch_preprocess(&mut self) -> usize {
        let mut level = 0;
        let mut contracted = vec![0u64; self.graph.num_nodes().div_ceil(64)];
        let mut all_num_shortcuts = 0;
        let mut all_num_contracted = 0;
        let mut nodes: Vec<usize> = (0..self.graph.num_nodes()).collect();
    
        loop {
            // Find independent set
            let indep_set = self.find_independent_set(&mut nodes);
            if indep_set.is_empty() {
                break;
            }

            // Contract part of independent set with low edge diff
            let (num_contracted, num_created) = self.contract_indep_set(&indep_set, level, &mut contracted, &mut nodes);
            all_num_shortcuts += num_created;
            all_num_contracted += num_contracted;
            
            println!("Created {} / Contracted {} at level {}", all_num_shortcuts, all_num_contracted, level);

            // Increase level for next independent set
            level += 1;
        }

        println!("#created: {}, #edges in new graph: {}", all_num_shortcuts, self.graph.num_edges());
        all_num_shortcuts
    }

    fn should_stall_forward(&self, node: usize) -> bool {
        let node_level = self.graph.node_at(node).level;
        let node_dist = match self.distances[node] {
            Some(d) => d,
            None => return false,
        };

        for edge in self.graph.incoming_edges(node) {
            let neighbor_level = self.graph.node_at(edge.0).level;

            // Only consider neighbors with higher level
            if neighbor_level > node_level {
                if let Some(alt_dist) = self.distances[edge.0] {
                    if alt_dist + edge.1 < node_dist {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn should_stall_backward(&self, node: usize) -> bool {
        let node_level = self.graph.node_at(node).level;
        let node_dist = match self.incoming_distances[node] {
            Some(d) => d,
            None => return false,
        };

        for edge in self.graph.outgoing_edges(node) {
            let neighbor_level = self.graph.node_at(edge.0).level;

            if neighbor_level > node_level {
                if let Some(alt_dist) = self.incoming_distances[edge.0] {
                    if alt_dist + edge.1 < node_dist {
                        return true;
                    }
                }
            }
        }
        false
    }
    
    fn find_independent_set(&self, nodes: &mut Vec<usize>) -> Vec<(isize, usize)> {
        let mut independent_set: Vec<(isize, usize)> = Vec::new();
        let mut blocked = vec![0u64; self.graph.num_nodes().div_ceil(64)];
        let mut blocked_nodes = Vec::new();
    
        for node in nodes.iter().map(|node| *node) {
            if blocked[node / 64] & (1 << (node % 64)) != 0 {
                blocked_nodes.push(node);
                continue;
            }

            let incoming_num = self.graph.incoming_edges(node).count() as isize; 
            let outgoing_num = self.graph.outgoing_edges(node).count() as isize; 

            independent_set.push((incoming_num * outgoing_num - incoming_num - outgoing_num, node));
            
            // Block its neighbors from being selected
            for (to, _) in self.graph.edges(node) {
                blocked[to / 64] |= 1 << (to % 64);
            }
        }

        *nodes = blocked_nodes;

        // Sort independent set
        independent_set.sort_by_key(|x| x.0);
        independent_set
    }
}

#[cfg(test)]
mod test_ch {
    use std::time::Instant;

    use super::CH;
    use crate::dijkstra::Dijkstra;
    use crate::graph::Graph;
    use crate::reader::parse_queries;

    #[test]
    fn test_ch_without_stall_on_demand() {
        // Load graph with CH levels
        let start = Instant::now();
        println!("Started parsing...");
        let graph = Graph::from_file("inputs/stgtregbz_ch.fmi").unwrap();
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
        let graph = Graph::from_file("inputs/stgtregbz_ch.fmi").unwrap();
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
    fn test_ch_own_preprocessing() {
        // Load graph
        let queries = parse_queries("inputs/queries.txt").unwrap();
        let graph = Graph::from_file("inputs/MV.fmi").unwrap();
        let mut ch = CH::new(graph);

        // Preprocess
        let start = Instant::now();
        println!("Started CH preprocessing...");
        ch.batch_preprocess();
        let duration = start.elapsed();
        println!("Preprocessed in {:.2?}", duration);
    
        let expected = [Some(210922),
            Some(211124),
            Some(212697),
            Some(211381),
            Some(210818),
            Some(213098),
            Some(210569),
            Some(211076),
            Some(212353),
            Some(210427),
            Some(212241),
            Some(212443),
            Some(214016),
            Some(212700),
            Some(212137),
            Some(214417),
            Some(211888),
            Some(212395),
            Some(213672),
            Some(211746),
            Some(214577),
            Some(214779),
            Some(216352),
            Some(215036),
            Some(214473),
            Some(216753),
            Some(214224),
            Some(214731),
            Some(216008),
            Some(214082),
            Some(215758),
            Some(215960),
            Some(217533),
            Some(216217),
            Some(215654),
            Some(217934),
            Some(215405),
            Some(215912),
            Some(217189),
            Some(215263)];

        for (i, (s, t)) in queries.iter().enumerate() {
            let start = Instant::now();
            assert_eq!(expected[i], ch.shortest_path(*s, *t, true));
            let duration = start.elapsed();
            println!("query took [{:.2?}]", duration);
        }
    }
}