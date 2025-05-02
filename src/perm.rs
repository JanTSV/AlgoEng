use crate::graph::{OffsetArray, Node, Edge};

pub struct Permutation {
    permutations: Vec<usize>
}

impl Permutation {
    pub fn by_level(graph: &OffsetArray) -> Self {
        let mut old_order: Vec<(usize, usize)> = Vec::new();

        // Get old order
        for (i, node) in graph.nodes.iter().enumerate() {
            old_order.push((i, node.level));
        }

        // Sort by level
        old_order.sort_by_key(|x| x.1);

        // Create permutation list
        // Index within list = old index, value at index = new index
        let mut permutations = Vec::new();
        for (new_id, _) in old_order {
            permutations.push(new_id);
        }

        Permutation { permutations }
    }

    pub fn permutate_graph(&self, graph: &OffsetArray) -> OffsetArray {
        let n = graph.nodes.len();
    
        // Build inverse mapping: old_id -> new_id
        let mut inverse = vec![0; n];
        for (new_id, &old_id) in self.permutations.iter().enumerate() {
            inverse[old_id] = new_id;
        }
    
        let mut nodes: Vec<Node> = Vec::with_capacity(n);
        let mut edges: Vec<Vec<Edge>> = vec![Vec::new(); n];
        let mut reverse_edges: Vec<Vec<Edge>> = vec![Vec::new(); n];
    
        for new_id in 0..n {
            let old_id = self.permutations[new_id];
    
            // Copy and insert node
            nodes.push(graph.node_at(old_id).clone());
    
            // Remap outgoing edges
            for edge in graph.outgoing_edges(old_id) {
                let mut edge = edge.clone();
                edge.to = inverse[edge.to];         // remap destination
                edges[new_id].push(edge);           // remap source
            }
    
            // Remap incoming edges
            for edge in graph.incoming_edges(old_id) {
                let mut edge = edge.clone();
                edge.to = inverse[edge.to];         // remap origin
                reverse_edges[new_id].push(edge);   // remap target
            }
        }
    
        OffsetArray::build_from(nodes, edges, reverse_edges)
    }

    pub fn from(&self, id: usize) -> usize {
        self.permutations[id]
    }
}