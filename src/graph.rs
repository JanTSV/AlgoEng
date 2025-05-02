use std::{fs::OpenOptions, error::Error};
use std::io::Write;


#[derive(Debug, Clone)]
pub struct Edge {
    pub to: usize,
    pub weight: u64,
    pub typ: u64,
    pub max_speed: i64,
    pub edge_id_a: Option<usize>,
    pub edge_id_b: Option<usize>,
}

impl Edge {
    pub fn new(to: usize, weight: u64, typ: u64, max_speed: i64, edge_id_a: Option<usize>, edge_id_b: Option<usize>) -> Self {
        Edge { to, weight, typ, max_speed, edge_id_a, edge_id_b }
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

    pub fn unflatten(&self) -> (Vec<Node>, Vec<Vec<Edge>>, Vec<Vec<Edge>>) {
        let num_nodes = self.nodes.len();
        let mut edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes];
        let mut reverse_edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes];

        for i in 0..self.nodes.len() {
            for edge in self.outgoing_edges(i) {
                edges[i].push(edge.clone());
            }

            for edge in self.incoming_edges(i) {
                reverse_edges[i].push(edge.clone());
            }
        }

        (self.nodes.clone(), edges, reverse_edges)
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

    pub fn to_file(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(filename)?;

        // Comment header
        for _ in 0..9 {
            writeln!(file, "# ")?;
        }

        // One empty line
        writeln!(file, "")?;

        // #nodes
        writeln!(file, "{}", self.nodes.len())?;

        // #edges
        writeln!(file, "{}", self.offsets.last().unwrap())?;

        // Print nodes: <ID> <OSMID> <Lat> <Lon> <Height> <Level>
        for (id, node) in self.nodes.iter().enumerate() {
            writeln!(file, "{} {} {} {} {} {}", id, node.osm_id, node.lat, node.lon, node.height, node.level)?;
        }

        // Print edges: <SrcID> <TrgID> <Weight> <Type> <MaxSpeed> <EdgeIdA> <EdgeIdB>
        for i in 0..self.offsets.len() - 1 {
            for edge in self.outgoing_edges(i) {
                let edge_id_a: String = match edge.edge_id_a {
                    Some(x) => x.to_string(),
                    None => "-1".to_string()
                };
                let edge_id_b: String = match edge.edge_id_b {
                    Some(x) => x.to_string(),
                    None => "-1".to_string()
                };

                writeln!(file, "{} {} {} {} {} {} {}", i, edge.to, edge.weight, edge.typ, edge.max_speed, edge_id_a, edge_id_b);
            }
        }

        // End empty line
        writeln!(file, "")?;

        Ok(())
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