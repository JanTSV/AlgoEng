use std::ops::Index;
use std::{fs::OpenOptions, error::Error};
use std::io::Write;
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
    pub level: usize
}

impl Node {
    pub fn new(level: usize) -> Self {
        Node { level }
    }
}

#[derive(Debug, Clone)]
pub struct Graph {
    // Nodes
    nodes: Vec<Node>,
    edges: Vec<Vec<(usize, u64, bool)>>,
}

impl Graph {
    pub fn from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let file = OpenOptions::new()
        .read(true)
        .open(filename)?;

        let mut reader = BufReader::new(file);

        let mut line = String::new();
        reader.read_line(&mut line)?;

        while line.starts_with('#') || line.trim().is_empty() {
            line.clear();
            reader.read_line(&mut line)?;
        }

        // Parse number of nodes and edges
        let num_nodes: usize = line.trim().parse().expect("Missing number of nodes");

        line.clear();
        reader.read_line(&mut line)?;
        let num_edges: usize = line.trim().parse().expect("Missing number of edges");

        let mut graph = Self::new(num_nodes, num_edges);

        // Parse nodes
        for _i in 0..num_nodes {
            line.clear();
            reader.read_line(&mut line)?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            
            if parts.len() < 5 {
                return Err(format!("Malformed node line {}, parts: {}", line, parts.len()).into());
            }
    
            let id: usize = parts[0].parse()?;
            assert_eq!(id, _i);
   
            let _osm_id: u64 = parts[1].parse()?;
            let _lat: f64 = parts[2].parse()?;
            let _lon: f64 = parts[3].parse()?;
            let _height: f64 = parts[4].parse()?;
            let level: usize = if parts.len() > 5 {
                parts[5].parse::<usize>().unwrap_or(usize::MAX)
            } else {
               usize::MAX
            };
            
            graph.nodes.push(Node::new(level));
        }

        // Parse edges
        for _ in 0..num_edges {
            line.clear();
            reader.read_line(&mut line)?;
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
            
            graph.edges[source].push((target, weight, true));
            graph.edges[target].push((source, weight, false));
        }

        assert_eq!(num_nodes, graph.nodes.len());
        assert_eq!(num_nodes, graph.edges.len());
        assert_eq!(num_edges, graph.num_edges());

        Ok(graph)
    }

    pub fn num_edges(&self) -> usize {
        self.edges
            .iter()
            .map(|edge| edge.iter().filter(|(_, _, forward)| *forward).count())
            .sum()
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    } 

    pub fn add_edge(&mut self, from: usize, to: usize, weight: u64) {
        self.edges[from].push((to, weight, true));
        self.edges[to].push((from, weight, false));
    }

    fn new(num_nodes: usize, num_edges: usize) -> Self {
        Graph { nodes: Vec::with_capacity(num_nodes), edges: vec![Vec::new(); num_nodes] }
    }

    pub fn node_at(&self, idx: usize) -> &Node {
        &self.nodes[idx]
    }

    pub fn node_at_mut(&mut self, idx: usize) -> &mut Node {
        &mut self.nodes[idx]
    }

    pub fn outgoing_edges(&self, idx: usize) -> impl Iterator<Item = (usize, u64)> + '_  {
        self.edges[idx]
            .iter()
            .rev()
            .filter_map(|(to, weight, dir)| dir.then_some((*to, *weight)))
    }

    pub fn incoming_edges(&self, idx: usize) -> impl Iterator<Item = (usize, u64)> + '_  {
        self.edges[idx]
            .iter()
            .rev()
            .filter_map(|(to, weight, dir)| (!dir).then_some((*to, *weight)))
    }

    pub fn edges(&self, idx: usize) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.edges[idx]
            .iter()
            .rev()
            .map(|(to, weight, _)| (*to, *weight))
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
            // TODO: writeln!(file, "{} {} {} {} {} {}", id, node.osm_id, node.lat, node.lon, node.height, node.level)?;
        }

        // TODO: Print edges: <SrcID> <TrgID> <Weight> <Type> <MaxSpeed> <EdgeIdA> <EdgeIdB>

        // End empty line
        writeln!(file, "")?;

        Ok(())
    }
}

#[cfg(test)]
mod test_offset_array {
    use super::Graph;

    #[test]
    fn test_offset_array_parse_toy() {
        let graph = Graph::from_file("inputs/toy.fmi").unwrap();
        assert_eq!(graph.nodes.len(), 5);
        assert_eq!(graph.num_edges(), 9);

        println!("{:?}", graph);
    }
    
    #[test]
    fn test_offset_array_parse_mv() {
        let graph = Graph::from_file("inputs/MV.fmi").unwrap();
        assert_eq!(graph.nodes.len(), 644199);
        assert_eq!(graph.num_edges(), 1305996);
    }
}

