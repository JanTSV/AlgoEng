use std::{fs::OpenOptions, error::Error};
use std::io::Write;
use std::fs::File;
use std::io::{BufRead, BufReader};


#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: u64,
    pub typ: u64,
    pub max_speed: i64,
    pub edge_id_a: Option<usize>,
    pub edge_id_b: Option<usize>,
}

impl Edge {
    pub fn new(from: usize, to: usize, weight: u64, typ: u64, max_speed: i64, edge_id_a: Option<usize>, edge_id_b: Option<usize>) -> Self {
        Edge { from, to, weight, typ, max_speed, edge_id_a, edge_id_b }
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
    offsets: Vec<usize>,
    reverse_offsets: Vec<usize>,
    
    // Nodes
    nodes: Vec<Node>,

    // Edges
    edges: Vec<Edge>,
    reverse_edges: Vec<Edge>
}

impl OffsetArray {
    pub fn from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        // Filter out comments and empty lines
        let mut lines = reader
            .lines()
            .map_while(Result::ok)
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty() && !line.starts_with('#'));

        // Parse number of nodes and edges
        let num_nodes: usize = lines.next().ok_or("Missing number of nodes")?.parse()?;
        let num_edges: usize = lines.next().ok_or("Missing number of edges")?.parse()?;

        let mut graph = Self::new(num_nodes, num_edges);

        // Parse nodes
        for _i in 0..num_nodes {
            let line = lines.next().ok_or("Missing edge line")?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            
            if parts.len() < 5 {
                return Err(format!("Malformed node line {}, parts: {}", line, parts.len()).into());
            }
    
            let id: usize = parts[0].parse()?;
            assert_eq!(id, _i);
   
            let osm_id: u64 = parts[1].parse()?;
            let lat: f64 = parts[2].parse()?;
            let lon: f64 = parts[3].parse()?;
            let height: f64 = parts[4].parse()?;
            let level: usize = if parts.len() > 5 {
                parts[5].parse::<usize>().unwrap_or(usize::MAX)
            } else {
               usize::MAX
            };
            
            graph.nodes.push(Node::new(osm_id, lat, lon, height, level));
        }

        // Parse edges
        for _ in 0..num_edges {
            let line = lines.next().ok_or("Missing edge line")?;
            let parts: Vec<&str> = line.split_whitespace().collect();
    
            if parts.len() < 5 {
                return Err(format!("Malformed edge line {}, parts: {}", line, parts.len()).into());
            }
    
            let source: usize = parts[0].parse()?;
            let target: usize = parts[1].parse()?;
            let weight: u64 = parts[2].parse()?;
            let typ: u64 = parts[3].parse()?;
            let max_speed: i64 = parts[4].parse()?;
            let edge_id_a: Option<usize> = if parts.len() > 5 {
                parts[5].parse().ok()
            } else {
                None
            };
            let edge_id_b: Option<usize> = if parts.len() > 6 {
                parts[6].parse().ok()
            } else {
                None
            };
            
            graph.edges.push(Edge::new(source, target, weight, typ, max_speed, edge_id_a, edge_id_b));
        }

        // Clone reverse edges with to and from swapped        
        graph.reverse_edges = graph.edges
            .iter()
            .map(|e| Edge::new(
                e.to,
                e.from,
                e.weight,
                e.typ,
                e.max_speed,
                e.edge_id_a,
                e.edge_id_b,
            ))
            .collect();

       graph.build_offsets();

        assert_eq!(num_nodes, graph.nodes.len());
        assert_eq!(num_edges, graph.edges.len());
        assert_eq!(num_edges, graph.reverse_edges.len());
        assert_eq!(num_nodes + 1, graph.offsets.len());
        assert_eq!(num_nodes + 1, graph.reverse_offsets.len());

        Ok(graph)
    }

    pub fn nodes_num(&self) -> usize {
        self.nodes.len()
    } 

    pub fn add_edge(&mut self, edge: Edge) {
        let rev_edge = Edge::new(edge.to, edge.from, edge.weight, edge.typ, edge.max_speed, edge.edge_id_a, edge.edge_id_b); 
        self.edges.push(edge);
        self.reverse_edges.push(rev_edge);
    }

    pub fn build_offsets(&mut self) {
        let num_nodes = self.nodes_num();

        // Clear old offsets
        self.offsets.clear();
        self.reverse_offsets.clear();

        // Sort edges and create offset array
        self.edges.sort_by_key(|edge| edge.from);

        let mut current_offset = 0;
        for node_id in 0..num_nodes {
            self.offsets.push(current_offset);
            while current_offset < self.edges.len() && self.edges[current_offset].from == node_id {
                current_offset += 1;
            }
        }
        self.offsets.push(current_offset);

        // Same for reverse edges.
        self.reverse_edges.sort_by_key(|edge| edge.from);

        let mut current_offset = 0;
        for node_id in 0..num_nodes {
            self.reverse_offsets.push(current_offset);
            while current_offset < self.reverse_edges.len() && self.reverse_edges[current_offset].from == node_id {
                current_offset += 1;
            }
        }
        self.reverse_offsets.push(current_offset);
    }

    fn new(num_nodes: usize, num_edges: usize) -> Self {
        OffsetArray { offsets: Vec::with_capacity(num_nodes + 1), reverse_offsets: Vec::with_capacity(num_nodes + 1), nodes: Vec::with_capacity(num_nodes), edges: Vec::with_capacity(num_edges), reverse_edges: Vec::with_capacity(num_edges) }
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

    pub fn num_edges(&self) -> usize {
        *self.offsets.last().unwrap()
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
        writeln!(file, "{}", self.num_edges())?;

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

                writeln!(file, "{} {} {} {} {} {} {}", edge.from, edge.to, edge.weight, edge.typ, edge.max_speed, edge_id_a, edge_id_b)?;
            }
        }

        // End empty line
        writeln!(file, "")?;

        Ok(())
    }
}

#[cfg(test)]
mod test_offset_array {
    use super::OffsetArray;

    #[test]
    fn test_offset_array_parse_toy() {
        let graph = OffsetArray::from_file("inputs/toy.fmi").unwrap();
        assert_eq!(graph.nodes.len(), 5);
        assert_eq!(graph.edges.len(), 9);
        assert_eq!(graph.offsets.len(), 6);
        assert_eq!(graph.reverse_offsets.len(), 6);

        println!("{:?}", graph);
    }
    
    #[test]
    fn test_offset_array_parse_mv() {
        let graph = OffsetArray::from_file("inputs/MV.fmi").unwrap();
        assert_eq!(graph.nodes.len(), 644199);
        assert_eq!(graph.edges.len(), 1305996);
        assert_eq!(graph.offsets.len(), 644200);
        assert_eq!(graph.reverse_offsets.len(), 644200);
    }
}

