use std::{fs::OpenOptions, error::Error};
use std::io::Write;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{Ordering, AtomicI32};
use std::sync::OnceLock;

pub type NodeId = u32;

#[derive(Debug)]
pub struct Node {
    osm_id: u64,
    lat: f32,
    lon: f32,
    level: u16
}

impl Node {
    pub fn new(osm_id: u64, lat: f32, lon: f32, level: u16) -> Self {
        Node { osm_id, lat, lon, level }
    }

    pub fn get_level(&self) -> &u16 {
        &self.level
    }

    pub fn set_level(&mut self, level: u16) {
        self.level = level
    }

    pub fn from_str(s: &str) -> Result<Self, Box<dyn Error>> {
        let mut split = s.trim().split(' ');
        let _id = split.next();
        let osm_id: u64 = split.next().expect("Error (node): OSM ID.").parse()?;
        let lat: f32 = split.next().expect("Error (node): Lat.").parse()?;
        let lon: f32 = split.next().expect("Error (node): Lon.").parse()?;
        let _height = split.next();
        let level: u16 = split
            .next()
            .and_then(|x| x.parse::<u16>().ok())
            .unwrap_or(u16::MAX);

        Ok(Self::new(osm_id, lat, lon, level))
    }
}

static EDGE_CREATION_COUNTER: OnceLock<AtomicI32> = OnceLock::new();
#[derive(Debug, Clone)]
pub struct Edge {
    target: NodeId,
    weight: u32,
    max_speed: i32,
    edge_id_a: i32,
    edge_id_b: i32,
    id: i32,
    dir: bool,
    typ: u8,
}

impl Edge {
    pub fn new(target: NodeId, weight: u32, typ: u8, max_speed: i32, edge_id_a: i32, edge_id_b: i32) -> Self {
        let counter = EDGE_CREATION_COUNTER.get_or_init(|| AtomicI32::new(0));
        counter.fetch_add(1, Ordering::Relaxed);
        Edge { target, weight, max_speed, edge_id_a, edge_id_b, id: Self::get_creation_count(), dir: true, typ }
    }

    fn get_creation_count() -> i32 {
        EDGE_CREATION_COUNTER.get().map(|c| c.load(Ordering::Relaxed)).unwrap_or(0)
    }

    pub fn reverse(&self, source: NodeId) -> (NodeId, Self) {
        let mut edge = self.clone();
        edge.dir = false;
        let target = edge.target;
        edge.target = source;
        (target, edge)
    }

    pub fn from_str(s: &str) -> Result<(NodeId, Self), Box<dyn Error>> {
        let mut split = s.trim().split(' ');
        let source: NodeId = split.next().expect("Error (edge): Source.").parse()?;
        let target: NodeId = split.next().expect("Error (edge): Target.").parse()?;
        let weight: u32 = split.next().expect("Error (edge): Weight.").parse()?;
        let typ: u8 = split.next().expect("Error (edge): Type.").parse()?;
        let max_speed: i32 = split.next().expect("Error (edge): Max speed.").parse()?;
        let edge_id_a: i32 = split
            .next()
            .and_then(|x| x.parse::<i32>().ok())
            .unwrap_or(-1);
        let edge_id_b: i32 = split
            .next()
            .and_then(|x| x.parse::<i32>().ok())
            .unwrap_or(-1);

        Ok((source, Self::new(target, weight, typ, max_speed, edge_id_a, edge_id_b)))
    }
}


#[derive(Debug)]
pub struct Graph {
    // Nodes
    nodes: Vec<Node>,

    // (to, weight, direction, (replaced_node, edge_id_a, edge_id_b))
    edges: Vec<Vec<Edge>>,
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

        // Parse nodes
        let mut nodes: Vec<Node> = Vec::with_capacity(num_nodes);
        for _i in 0..num_nodes {
            line.clear();
            reader.read_line(&mut line)?;
            
            nodes.push(Node::from_str(&line).expect("Could not parse node"));
        }

        // Parse edges
        let mut edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes];
        for _ in 0..num_edges {
            line.clear();
            reader.read_line(&mut line)?;

            let (source, edge) = Edge::from_str(&line).expect("Could not parse edge");
            let (rev_source, rev_edge) = edge.reverse(source);
            edges[source as usize].push(edge);
            edges[rev_source as usize].push(rev_edge);
        }

        // nodes.shrink_to_fit();

        Ok(Self::new(nodes, edges))
    }

    pub fn num_edges(&self) -> usize {
        self.edges
            .iter()
            .map(|edge| edge.iter().filter(|&edge| edge.dir).count())
            .sum()
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    } 

    pub fn add_edge(&mut self, source: NodeId, edge: Edge) {
        let (rev_source, rev_edge) = edge.reverse(source);
        self.edges[source as usize].push(edge);
        self.edges[rev_source as usize].push(rev_edge);
    }

    fn new(nodes: Vec<Node>, edges: Vec<Vec<Edge>>) -> Self {
        assert_eq!(nodes.len(), edges.len());
        Graph { nodes, edges }
    }

    pub fn node_at(&self, idx: NodeId) -> &Node {
        &self.nodes[idx as usize]
    }

    pub fn node_at_mut(&mut self, idx: NodeId) -> &mut Node {
        &mut self.nodes[idx as usize]
    }

    pub fn outgoing_edges(&self, idx: NodeId) -> impl Iterator<Item = (NodeId, u32)> + '_ + DoubleEndedIterator {
        self.edges[idx as usize]
            .iter()
            .filter_map(|edge| edge.dir.then_some((edge.target, edge.weight)))
    }
    
    pub fn incoming_edges(&self, idx: NodeId) -> impl Iterator<Item = (NodeId, u32)> + '_ + DoubleEndedIterator  {
        self.edges[idx as usize]
            .iter()
            .filter_map(|edge| (!edge.dir).then_some((edge.target, edge.weight)))
    }

    pub fn outgoing_edges_with_id(&self, idx: NodeId) -> impl Iterator<Item = (NodeId, u32, i32)> + '_ + DoubleEndedIterator {
        self.edges[idx as usize]
            .iter()
            .filter_map(|edge| edge.dir.then_some((edge.target, edge.weight, edge.id)))
    }
    
    pub fn incoming_edges_with_id(&self, idx: NodeId) -> impl Iterator<Item = (NodeId, u32, i32)> + '_ + DoubleEndedIterator  {
        self.edges[idx as usize]
            .iter()
            .filter_map(|edge| (!edge.dir).then_some((edge.target, edge.weight, edge.id)))
    }

    pub fn edges(&self, idx: NodeId) -> impl Iterator<Item = (NodeId, u32, bool)> + '_ + DoubleEndedIterator {
        self.edges[idx as usize]
            .iter()
            .map(|edge| (edge.target, edge.weight, edge.dir))
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
        writeln!(file, "# ")?;

        // #nodes
        writeln!(file, "{}", self.num_nodes())?;

        // #edges
        writeln!(file, "{}", self.num_edges())?;

        // nodes
        for (i, node) in self.nodes.iter().enumerate() {
            // <ID> <OSMID> <Lat> <Lon> <Height> <Level>
            writeln!(file, "{} {} {} {} 0 {}", i, node.osm_id, node.lat, node.lon, node.level)?;
        }

        // edges
        for (node, edges) in self.edges.iter().enumerate() {
            for edge in edges {
                if !edge.dir {
                    continue;
                }

                // <SrcID> <TrgID> <Weight> <Type> <MaxSpeed> <EdgeIdA> <EdgeIdB>
                writeln!(file, "{} {} {} {} {} {} {}", node, edge.target, edge.weight, edge.typ, edge.max_speed, edge.edge_id_a, edge.edge_id_b)?;
            }
        }

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

