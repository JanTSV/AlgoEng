use std::collections::BinaryHeap;
use std::error::Error;
use std::ops::Div;
use std::time::Instant;
use std::sync::{Arc, RwLock, RwLockReadGuard};

use rayon::prelude::*;

use crate::graph::{Graph, NodeId, Edge};
use crate::dijkstra::{Distance, Dijkstra};

struct Shortcut {
    from: NodeId,
    to: NodeId,
    weight: u32,
    edge_id_a: i32,
    edge_id_b: i32,
}

impl Shortcut {
    pub fn new(from: NodeId, to: NodeId, weight: u32, edge_id_a: i32, edge_id_b: i32) -> Self {
        Shortcut { from, to, weight, edge_id_a, edge_id_b }
    }
}

pub struct CH {
    graph: Arc<RwLock<Graph>>,

    // s -> t
    distances: Vec<Option<u32>>,
    heap: BinaryHeap<Distance>,
    visited: Vec<NodeId>,

    // t -> s
    incoming_distances: Vec<Option<u32>>,
    incoming_heap: BinaryHeap<Distance>,

    optimized: Vec<u64>,
    incoming_optimized: Vec<u64>,
}

impl CH {
    pub fn new(graph: Graph) -> Self {
        let n = graph.num_nodes();
        CH { graph: Arc::new(RwLock::new(graph)), 
             distances: vec![None; n], 
             heap: BinaryHeap::new(), 
             visited: Vec::new(),
             incoming_distances: vec![None; n], 
             incoming_heap: BinaryHeap::new(),
             optimized: vec![0u64; n.div_ceil(64)],
             incoming_optimized: vec![0u64; n.div_ceil(64)],
        }
    }

    pub fn get_graph(&self) -> RwLockReadGuard<'_, Graph> {
        let graph = self.graph.read().unwrap();
        graph
    }

    pub fn shortest_path(&mut self, s: NodeId, t: NodeId, stall_on_demand: bool) -> Option<u32> {
        // Cleanup of previous run
        while let Some(node) = self.visited.pop() {
            self.distances[node as usize] = None;
            self.incoming_distances[node as usize] = None;
            self.optimized[node as usize / 64] &= !(1 << (node % 64));
            self.incoming_optimized[node as usize / 64] &= !(1 << (node % 64));
        }
        let graph = self.graph.read().unwrap();

        let mut distance: Option<u32> = None;
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
                if self.optimized[id as usize / 64] & (1 << (id % 64)) != 0 {
                    continue;
                }

                // Stall-on-demand
                if stall_on_demand && self.should_stall_forward(id) {
                    continue;
                }

                // Other dijkstra also reached this node
                if self.incoming_optimized[id as usize / 64] & (1 << (id % 64)) != 0 {
                    let new_dist = weight + unsafe { self.incoming_distances.get_unchecked(id as usize).unwrap() };
                    distance = Some(distance.map_or(new_dist, |dist| dist.min(new_dist)));
                }
    
                self.optimized[id as usize / 64] |= 1 << (id % 64);

                for edge in graph.outgoing_edges(id).rev() {
                    //assert!(*graph.node_at(edge.0).get_level() != u16::MAX);
                    //assert!(*graph.node_at(id).get_level() != u16::MAX);
                    if unsafe { self.distances.get_unchecked(edge.0 as usize) }.is_none_or(|curr| weight + edge.1 < curr) && 
                       graph.node_at(edge.0).level > graph.node_at(id).level {
                        self.distances[edge.0 as usize] = Some(weight + edge.1);
                        self.heap.push(Distance::new(weight + edge.1, edge.0));
                        self.visited.push(edge.0);
                    }
                }
            }

            // Dijkstra from t
            if let Some(Distance { weight, id }) = self.incoming_heap.pop() {
                if self.incoming_optimized[id as usize / 64] & (1 << (id % 64)) != 0 {
                    continue;
                }

                // Stall-on-demand
                if stall_on_demand && self.should_stall_backward(id) {
                    continue;
                }

                // Other dijkstra also reached this node
                if self.optimized[id as usize / 64] & (1 << (id % 64)) != 0 {
                    let new_dist = weight + unsafe { self.distances.get_unchecked(id as usize).unwrap() };
                    distance = Some(distance.map_or(new_dist, |dist| dist.min(new_dist)));
                }
    
                self.incoming_optimized[id as usize / 64] |= 1 << (id % 64);


                for edge in graph.incoming_edges(id).rev() {
                    //assert!(*graph.node_at(edge.0).get_level() != u16::MAX);
                    //assert!(*graph.node_at(id).get_level() != u16::MAX);
                    if unsafe { self.incoming_distances.get_unchecked(edge.0 as usize) }.is_none_or(|curr| weight + edge.1 < curr) && 
                       graph.node_at(edge.0).level > graph.node_at(id).level {
                        self.incoming_distances[edge.0 as usize] = Some(weight + edge.1);
                        self.incoming_heap.push(Distance::new(weight + edge.1, edge.0));
                        self.visited.push(edge.0);
                    }
                }
            }
        }

        distance
    }

    fn calc_shortcuts(&self, dijkstra: &mut Dijkstra, node: NodeId, contracted: &[u64]) -> Vec<Shortcut> {
        // (from, to, weight, edge_id_a, edge_id_b)
        let mut shortcuts: Vec<Shortcut> = Vec::new();
        let graph = self.graph.read().unwrap();

        for (pred_id, pred_weight, edge_id_b) in graph
            .incoming_edges_with_id(node) {

            if contracted[pred_id as usize / 64] & (1 << (pred_id % 64)) != 0 {
                continue;
            }

            for (succ_id, succ_weight, edge_id_a) in graph
                .outgoing_edges_with_id(node) {
                    
                if contracted[succ_id as usize / 64] & (1 << (succ_id % 64)) != 0 {
                    continue;
                }

                let direct_distance = pred_weight + succ_weight;
                let shortest_distance = dijkstra.shortest_path_consider_contraction(pred_id, succ_id, contracted).expect("this should never happen");
                if shortest_distance >= direct_distance {
                    shortcuts.push(Shortcut::new(pred_id, succ_id, direct_distance, edge_id_a, edge_id_b));
                }
            }
        }

        shortcuts
    }

    fn calc_shortcuts_parallel(&self, node: NodeId, contracted: &[u64]) -> Vec<Shortcut> {
        let incoming_edges: Vec<_> = {
            let graph = self.graph.read().unwrap();
                graph
                .incoming_edges_with_id(node)
                .collect()
        };

        let chunk_size = incoming_edges.len().div_ceil(rayon::current_num_threads());

        incoming_edges
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                let graph = self.graph.read().unwrap();
                let mut dijkstra = Dijkstra::new(&graph);

                let mut local_shortcuts = Vec::new();

                for (pred_id, pred_weight, edge_id_b) in chunk {
                    if contracted[*pred_id as usize / 64] & (1 << (pred_id % 64)) != 0 {
                        continue;
                    }

                    for (succ_id, succ_weight, edge_id_a) in graph.outgoing_edges_with_id(node) {
                        if contracted[succ_id as usize / 64] & (1 << (succ_id % 64)) != 0 {
                            continue;
                        }

                        let direct_distance = pred_weight + succ_weight;
                        let shortest_distance = dijkstra
                            .shortest_path_consider_contraction(*pred_id, succ_id, contracted)
                            .expect("this should never happen");

                        if shortest_distance >= direct_distance {
                            local_shortcuts.push(Shortcut::new(*pred_id, succ_id, direct_distance, edge_id_a, *edge_id_b));
                        }
                    }
                }
                local_shortcuts
            })
            .collect()
    }

    fn contract_indep_set(
        &mut self,
        indep_set: &Vec<(isize, NodeId)>,
        level: u16,
        contracted: &mut [u64]
    ) {
        //let start = Instant::now();
        let chunk_size = indep_set.len().div_ceil(rayon::current_num_threads());
        let results: Vec<(NodeId, Vec<Shortcut>)> = indep_set
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                let graph = self.graph.read().unwrap();
                let mut dijkstra = Dijkstra::new(&graph);
                chunk.iter().map(|&(_, node)| {
                    let shortcuts = self.calc_shortcuts(&mut dijkstra, node, contracted);
                    (node, shortcuts)
                }).collect::<Vec<_>>()
            })
            .collect();

        //println!("Threads: {:.2?}", start.elapsed());
    
        let mut graph = self.graph.write().unwrap();
        for (node, shortcuts) in results {
            for Shortcut { from, to, weight, edge_id_a, edge_id_b } in shortcuts {
                graph.add_edge(from, Edge::new(to, weight, 0, -1, edge_id_a, edge_id_b));
            }
    
            graph.node_at_mut(node).set_level(level);
            contracted[node as usize / 64] |= 1 << (node % 64);

        }
    }

    pub fn batch_preprocess(&mut self) {
        let mut contracted = {
            let graph = self.graph.read().unwrap();
            vec![0u64; graph.num_nodes().div_ceil(64)]
        };
        let mut nodes: Vec<NodeId> = {
            let graph = self.graph.read().unwrap();
            (0..graph.num_nodes() as u32).collect()
        };
    
        for level in 0.. {
            // Find independent set
            //let start = Instant::now();
            let indep_set = self.find_independent_set(&mut nodes, &contracted);
            //println!("find_independent_set took {:.2?}", start.elapsed());
            if indep_set.is_empty() {
                break;
            }

            // Contract part of independent set with low edge diff
            //let start = Instant::now();
            self.contract_indep_set(&indep_set, level, &mut contracted);
            //println!("contract_indep_set took {:.2?}", start.elapsed());
            
            dbg!(level);
        }
    }

    fn should_stall_forward(&self, node: NodeId) -> bool {
        let graph = self.graph.read().unwrap();
        let node_level = graph.node_at(node).level;
        let node_dist = match self.distances[node as usize] {
            Some(d) => d,
            None => return false,
        };

        for edge in graph.incoming_edges(node) {
            let neighbor_level = graph.node_at(edge.0).level;

            // Only consider neighbors with higher level
            if neighbor_level > node_level {
                if let Some(alt_dist) = self.distances[edge.0 as usize] {
                    if alt_dist + edge.1 < node_dist {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn should_stall_backward(&self, node: NodeId) -> bool {
        let graph = self.graph.read().unwrap();
        let node_level = graph.node_at(node).level;
        let node_dist = match self.incoming_distances[node as usize] {
            Some(d) => d,
            None => return false,
        };

        for edge in graph.outgoing_edges(node) {
            let neighbor_level = graph.node_at(edge.0).level;

            if neighbor_level > node_level {
                if let Some(alt_dist) = self.incoming_distances[edge.0 as usize] {
                    if alt_dist + edge.1 < node_dist {
                        return true;
                    }
                }
            }
        }
        false
    }
    
    fn find_independent_set(&self, nodes: &mut Vec<NodeId>, contracted: &[u64]) -> Vec<(isize, NodeId)> {
        let mut independent_set: Vec<(isize, NodeId)> = Vec::new();
        let graph = self.graph.read().unwrap();
        let mut blocked = vec![0u64; graph.num_nodes().div_ceil(64)];
        let mut blocked_nodes = Vec::new();

        dbg!(nodes.len());
    
        for node in nodes.iter().map(|node| *node) {
            if blocked[node as usize / 64] & (1 << (node % 64)) != 0 {
                blocked_nodes.push(node);
                continue;
            }

            let incoming_num = graph.incoming_edges(node).filter(|(to, _)| (contracted[*to as usize / 64]) & (1 << (*to % 64)) == 0).count() as isize;
            let outgoing_num = graph.outgoing_edges(node).filter(|(to, _)| (contracted[*to as usize / 64]) & (1 << (*to % 64)) == 0).count() as isize;

            independent_set.push((incoming_num * outgoing_num - incoming_num - outgoing_num, node));
            
            // Block its neighbors from being selected
            for (to, _, _) in graph.edges(node) {
                blocked[to as usize / 64] |= 1 << (to % 64);
            }
        }

        // Sort independent set
        independent_set.sort_by_key(|x| x.0);

        let zero_or_less = independent_set.partition_point(|x| x.0 <= 0);

        // Only take subset with low edge diff
        let n = if zero_or_less > 0 {
            zero_or_less.max(independent_set.len().div_ceil(16))
        } else {
            independent_set.len().div_ceil(8)
        };

        dbg!(n);

         for (_, node) in &independent_set[n..] {
            blocked_nodes.push(*node);
        }

        independent_set.truncate(n);

        // Save blocked nodes for next call
        *nodes = blocked_nodes;

        independent_set
    }
}

#[cfg(test)]
mod test_ch {
    use std::time::Instant;

    use super::CH;
    use crate::dijkstra::Dijkstra;
    use crate::graph::{Graph, NodeId};
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
        const START: NodeId = 377371;
        const TARGET: NodeId = 754742;
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
        const START: NodeId = 377371;
        const TARGET: NodeId = 754742;
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
        let queries = parse_queries("inputs/querries.txt").unwrap();
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