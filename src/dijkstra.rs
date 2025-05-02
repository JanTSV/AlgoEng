use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::graph::OffsetArray;

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
    graph: &'a OffsetArray,
    weights: Vec<Option<u64>>,
    heap: BinaryHeap<Distance>,
    visited: Vec<usize>
}

impl<'a> Dijkstra<'a> {
    pub fn new(graph: &'a OffsetArray) -> Self {
        Dijkstra { graph, weights: vec![None; graph.nodes.len()], heap: BinaryHeap::new(), visited: Vec::new() }
    }

    pub fn unsafe_new(graph: *const OffsetArray) -> Self {
        let graph = unsafe { &*graph };
        Self::new(graph)
    }

    pub fn shortest_path(&mut self, s: usize, t: usize) -> Option<u64> {
        // Cleanup of previous run
        while let Some(node) = self.visited.pop() {
            self.weights[node] = None;
        }
        self.heap.clear();

        // Push start to heap and set dist to 0
        self.heap.push(Distance::new(0, s));
        self.weights[s] = Some(0);
        self.visited.push(s);

        while let Some(Distance { weight, id }) = self.heap.pop() {
            if id == t {
                return Some(weight);
            }

            for edge in self.graph.outgoing_edges(id) {
                if self.weights[edge.to].is_none_or(|curr| weight + edge.weight < curr) {
                    self.weights[edge.to] = Some(weight + edge.weight);
                    self.heap.push(Distance::new(weight + edge.weight, edge.to));
                    self.visited.push(edge.to);
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod test_dijkstra {
    use std::time::Instant;

    use crate::reader::{parse_graph, parse_queries};

    use super::Dijkstra;

    #[test]
    fn test_dijkstra_in_mv() {
        let queries = parse_queries("inputs/queries.txt").unwrap();
        let graph = parse_graph("inputs/MV.fmi").unwrap();
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
}