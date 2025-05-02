
#[derive(Debug, Clone)]
pub struct Edge {
    pub to: usize,
    pub weight: u64,
    pub edge_id_a: Option<usize>,
    pub edge_id_b: Option<usize>,
}

impl Edge {
    pub fn new(to: usize, weight: u64, edge_id_a: Option<usize>, edge_id_b: Option<usize>) -> Self {
        Edge { to, weight, edge_id_a, edge_id_b }
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub osm_id: u64,
    pub lat: f64,
    pub lon: f64,
    pub height: f64,
    pub level: usize
}

impl Node {
    pub fn new(osm_id: u64, lat: f64, lon: f64, height: f64, level: usize) -> Self {
        Node { osm_id, lat, lon, height, level }
    }
}

#[derive(Debug, Clone)]
pub struct OffsetArray {
    // Offsets
    pub offsets: Vec<usize>,
    pub reverse_offsets: Vec<usize>,
    
    // Nodes
    pub nodes: Vec<Node>,

    // Edges
    pub edges: Vec<Edge>,
    pub reverse_edges: Vec<Edge>
}

impl OffsetArray {
    pub fn build_from(nodes: Vec<Node>, edges: Vec<Vec<Edge>>, reverse_edges: Vec<Vec<Edge>>) -> Self {
        assert_eq!(nodes.len(), edges.len());
        assert_eq!(nodes.len(), reverse_edges.len());

        let (edges, offsets) = Self::flatten(edges);
        assert_eq!(offsets.len(), nodes.len() + 1);
        let (reverse_edges, reverse_offsets) = Self::flatten(reverse_edges);
        assert_eq!(reverse_offsets.len(), nodes.len() + 1);
        OffsetArray { offsets, reverse_offsets, nodes, edges, reverse_edges }
    }

    pub fn node_at(&self, idx: usize) -> &Node {
        &self.nodes[idx]
    }

    pub fn node_at_mut(&mut self, idx: usize) -> &mut Node {
        &mut self.nodes[idx]
    }

    pub fn outgoing_edges(&self, idx: usize) -> &[Edge] {
         &self.edges[self.offsets[idx]..self.offsets[idx + 1]]
    }

    pub fn incoming_edges(&self, idx: usize) -> &[Edge] {
         &self.reverse_edges[self.reverse_offsets[idx]..self.reverse_offsets[idx + 1]]
    }

    fn flatten(edge_list: Vec<Vec<Edge>>) -> (Vec<Edge>, Vec<usize>) {
        let mut flat_edges: Vec<Edge> = Vec::new();
        let mut offsets: Vec<usize> = Vec::new();
        let mut current_offset: usize = 0;

        offsets.push(current_offset);

        for edges in edge_list {
            current_offset += edges.len();
            offsets.push(current_offset);
            flat_edges.extend(edges);
        }

        (flat_edges, offsets)
    }
}