use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::{HashSet, HashMap};

use crate::graph::Graph;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Distance {
    pub weight: u64,
    pub id: usize,
}

impl Distance {
    pub fn new(weight: u64, id: usize) -> Self {
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

pub struct Dijkstra<'a> {
    graph: &'a Graph,
    weights: Vec<Option<u64>>,
    heap: BinaryHeap<Distance>,
    visited: Vec<usize>,

    old_start: Option<usize>,
    optimized: Vec<u64>,
}

impl<'a> Dijkstra<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        Dijkstra { graph, weights: vec![None; graph.num_nodes()], heap: BinaryHeap::new(), visited: Vec::new(), old_start: None, optimized: vec![0u64; graph.num_nodes().div_ceil(64)] }
    }

    //pub fn _shortest_path_consider_contraction(&mut self, s: usize, t: usize, contracted: &[u64]) -> Option<u64> {
    //    // Cleanup of previous run
    //    while let Some(node) = self.visited.pop() {
    //        self.weights[node] = None;
    //        self.optimized[node / 64] &= !(1 << (node % 64));
    //    }
    //    self.heap.clear();
    //
    //    // Push start to heap and set dist to 0
    //    self.heap.push(Distance::new(0, s));
    //    self.weights[s] = Some(0);
    //    self.visited.push(s);
    //
    //    while let Some(Distance { weight, id }) = self.heap.pop() {
    //        if self.optimized[id / 64] & (1 << (id % 64)) != 0 {
    //            continue;
    //        }
    //
    //        self.optimized[id / 64] |= 1 << (id % 64);
    //
    //        if id == t {
    //            return Some(weight);
    //        }
    //
    //        for edge in self.graph.outgoing_edges(id) {
    //            if contracted[edge.0 / 64] & (1 << (edge.0 % 64)) != 0 {
    //                continue;
    //            }
    //
    //            if self.weights[edge.0].is_none_or(|curr| weight + edge.1 < curr) {
    //                self.weights[edge.0] = Some(weight + edge.1);
    //                self.heap.push(Distance::new(weight + edge.1, edge.0));
    //                self.visited.push(edge.0);
    //            }
    //        }
    //    }
    //
    //    None
    //}

    pub fn shortest_path_consider_contraction(&mut self, s: usize, t: usize, contracted: &[u64]) -> Option<u64> {
        if let Some(weight) = self.reset(s, t) {
            return Some(weight);
        }
        
        while let Some(Distance { weight, id }) = self.heap.pop() {
            if self.optimized[id / 64] & (1 << (id % 64)) != 0 {
                continue;
            }

            self.optimized[id / 64] |= 1 << (id % 64);

            for edge in self.graph.outgoing_edges(id).rev() {
                if contracted[edge.0 / 64] & (1 << (edge.0 % 64)) != 0 {
                    continue;
                }

                if self.weights[edge.0].is_none_or(|curr| weight + edge.1 < curr) {
                    self.weights[edge.0] = Some(weight + edge.1);
                    self.heap.push(Distance::new(weight + edge.1, edge.0));
                    self.visited.push(edge.0);
                }
            }
            
            if id == t {
                return Some(weight);
            }
        }

        None
    }

    // pub fn _shortest_path_consider_contraction(&mut self, s: usize, ts: &[usize], contracted: &[u64]) -> HashMap<usize, u64> {
    //     let target_set: HashSet<usize> = ts.iter().cloned().collect();
    //     let mut found: HashMap<usize, u64> = HashMap::new();
    //     let mut optimized = vec![0u64; self.graph.num_nodes().div_ceil(64)];
    // 
    //     // Cleanup of previous run
    //     while let Some(node) = self.visited.pop() {
    //         self.weights[node] = None;
    //     }
    //     self.heap.clear();
    // 
    //     // Push start to heap and set dist to 0
    //     self.heap.push(Distance::new(0, s));
    //     self.weights[s] = Some(0);
    //     self.visited.push(s);
    // 
    //     while let Some(Distance { weight, id }) = self.heap.pop() {
    //         // Skip already optimized nodes
    //         if optimized[id / 64] & (1 << (id % 64)) != 0 {
    //             continue;
    //         }
    // 
    //         // Mark as optimized
    //         optimized[id / 64] |= 1 << (id % 64);
    // 
    //         if target_set.contains(&id) && !found.contains_key(&id) {
    //             found.insert(id, weight);
    // 
    //             if found.len() == target_set.len() {
    //                 return found;
    //             }
    //         }
    // 
    //         for edge in self.graph.outgoing_edges(id) {
    //             if contracted[edge.0 / 64] & (1 << (edge.0 % 64)) != 0 {
    //                 continue;
    //             }
    // 
    //             if self.weights[edge.0].is_none_or(|curr| weight + edge.1 < curr) {
    //                 self.weights[edge.0] = Some(weight + edge.1);
    //                 self.heap.push(Distance::new(weight + edge.1, edge.0));
    //                 self.visited.push(edge.0);
    //             }
    //         }
    //     }
    // 
    //     found
    // }

    fn reset(&mut self, s: usize, t: usize) -> Option<u64> {
        let mut reset = true;

        if let Some(old_start) = self.old_start {
            if old_start == s {
                reset = false;
                if self.optimized[t / 64] & (1 << (t % 64)) != 0 {
                    return self.weights[t];
                }
            }
        }

        self.old_start = Some(s);

        // Cleanup of previous run
        if reset {
            while let Some(node) = self.visited.pop() {
                self.weights[node] = None;
                self.optimized[node / 64] &= !(1 << (node % 64));
            }
            self.heap.clear();
    
            // Push start to heap and set dist to 0
            self.heap.push(Distance::new(0, s));
            self.weights[s] = Some(0);
            self.visited.push(s);
        }

        return None;
    }

    pub fn shortest_path(&mut self, s: usize, t: usize) -> Option<u64> {
        if let Some(weight) = self.reset(s, t) {
            return Some(weight);
        }

        while let Some(Distance { weight, id }) = self.heap.pop() {
            if self.optimized[id / 64] & (1 << (id % 64)) != 0 {
                continue;
            }

            self.optimized[id / 64] |= 1 << (id % 64);

            for edge in self.graph.outgoing_edges(id) {
                if self.weights[edge.0].is_none_or(|curr| weight + edge.1 < curr) {
                    self.weights[edge.0] = Some(weight + edge.1);
                    self.heap.push(Distance::new(weight + edge.1, edge.0));
                    self.visited.push(edge.0);
                }
            }
            
            if id == t {
                return Some(weight);
            }
        }

        None
    }
}

#[cfg(test)]
mod test_dijkstra {
    use std::time::Instant;

    use crate::{reader::parse_queries, graph::Graph};

    use super::Dijkstra;

    #[test]
    fn test_dijkstra_in_mv() {
        let queries = parse_queries("inputs/querries.txt").unwrap();
        let graph = Graph::from_file("inputs/MV.fmi").unwrap();
        let mut dijkstra = Dijkstra::new(&graph);

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
            assert_eq!(expected[i], dijkstra.shortest_path(*s, *t));
            let duration = start.elapsed();
            println!("query took [{:.2?}]", duration);
        }
    }

    #[test]
    fn test_dijkstra_in_toy() {
        let graph = Graph::from_file("inputs/toy.fmi").unwrap();
        let mut dijkstra = Dijkstra::new(&graph);

        //Distance from 2 to 3: 5
        //Distance from 2 to 4: 4
        assert_eq!(Some(4), dijkstra.shortest_path(2, 4));
        assert_eq!(Some(5), dijkstra.shortest_path(2, 3));

    }
}