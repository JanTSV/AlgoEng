use std::{fs::OpenOptions, error::Error};
use std::io::Write;
use std::io::{BufRead, BufReader};


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

    // (to, weight, direction, (replaced_node, edge_id_a, edge_id_b))
    edges: Vec<Vec<(usize, u64, bool, Option<(usize, usize, usize)>)>>,
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

        let mut graph = Self::new(num_nodes);

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
            //let typ: u64 = parts[3].parse()?;
            //let max_speed: i64 = parts[4].parse()?;
            //let edge_id_a: Option<usize> = if parts.len() > 5 {
            //    parts[5].parse().ok()
            //} else {
            //    None
            //};
            //let edge_id_b: Option<usize> = if parts.len() > 6 {
            //    parts[6].parse().ok()
            //} else {
            //    None
            //};
            
            graph.edges[source].push((target, weight, true, None));
            graph.edges[target].push((source, weight, false, None));
        }

        assert_eq!(num_nodes, graph.nodes.len());
        assert_eq!(num_nodes, graph.edges.len());
        assert_eq!(num_edges, graph.num_edges());

        Ok(graph)
    }

    pub fn num_edges(&self) -> usize {
        self.edges
            .iter()
            .map(|edge| edge.iter().filter(|(_, _, forward, _)| *forward).count())
            .sum()
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    } 

    pub fn add_edge(&mut self, from: usize, to: usize, weight: u64, replaced: Option<(usize, usize, usize)>) {
        self.edges[from].push((to, weight, true, replaced));
        self.edges[to].push((from, weight, false, None));
    }

    fn new(num_nodes: usize) -> Self {
        Graph { nodes: Vec::with_capacity(num_nodes), edges: vec![Vec::new(); num_nodes] }
    }

    pub fn node_at(&self, idx: usize) -> &Node {
        &self.nodes[idx]
    }

    pub fn node_at_mut(&mut self, idx: usize) -> &mut Node {
        &mut self.nodes[idx]
    }

    pub fn outgoing_edges(&self, idx: usize) -> impl Iterator<Item = (usize, u64)> + '_ + DoubleEndedIterator {
        self.edges[idx]
            .iter()
            .filter_map(|(to, weight, dir, _)| dir.then_some((*to, *weight)))
    }

    fn shortcuts(&self, idx: usize) -> impl Iterator<Item = (usize, u64, usize, usize, usize)> + '_ + DoubleEndedIterator {
        self.edges[idx]
            .iter()
            .filter_map(|(to, weight, dir, replaced)| {
                if *dir {
                    replaced.map(|(node, edge_id_a, edge_id_b)| {
                        (*to, *weight, node, edge_id_a, edge_id_b)
                    })
                } else {
                    None
                }
            })
    }

    fn find_edge(&self, idx: usize, edge_id: usize, dir: bool) -> Option<usize> {
        let mut edge_ctr = 0_usize;
        for (i, edge) in self.edges[idx].iter().enumerate() {
            if edge.2 == dir {
                edge_ctr += 1;
            }

            if edge_id + 1 == edge_ctr {
                assert_eq!(edge.2, dir);
                return Some(i);
            }
        }
        None
    }

    fn calc_outgoing_edge_id(&self, idx: usize, edge_id_a: usize) -> usize {
        // Must be outgoing edge
        self.edge_count_until(idx) + self.find_edge(idx, edge_id_a, true).unwrap()
    }

    fn edge_count_until(&self, idx: usize) -> usize {
        self.edges
            .iter()
            .take(idx - 1)
            .map(|edge| edge.iter().filter(|(_, _, forward, _)| *forward).count())
            .sum::<usize>()
    }

    fn calc_incoming_edge_id(&self, idx: usize, edge_id_b: usize) -> usize {
        let incoming_edge = &self.edges[idx][self.find_edge(idx, edge_id_b, false).unwrap()];

        self.edge_count_until(incoming_edge.0) + self.edges[incoming_edge.0]
            .iter()
            .position(|(to, ..)| *to == idx )
            .unwrap()
    }

    pub fn incoming_edges(&self, idx: usize) -> impl Iterator<Item = (usize, u64)> + '_ + DoubleEndedIterator  {
        self.edges[idx]
            .iter()
            .filter_map(|(to, weight, dir, _)| (!dir).then_some((*to, *weight)))
    }

    pub fn edges(&self, idx: usize) -> impl Iterator<Item = (usize, u64)> + '_ + DoubleEndedIterator {
        self.edges[idx]
            .iter()
            .map(|(to, weight, _, _)| (*to, *weight))
    }

    pub fn to_file(&self, filename: &str, old_file: &str) -> Result<(), Box<dyn Error>> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(filename)?;

        let old_file = OpenOptions::new()
            .read(true)
            .open(old_file)?;

        let mut old_reader = BufReader::new(old_file);

        let mut old_line = String::new();
        old_reader.read_line(&mut old_line)?;
        
        while old_line.starts_with('#') || old_line.trim().is_empty() {
            old_line.clear();
            old_reader.read_line(&mut old_line)?;
        }

        // Comment header
        for _ in 0..9 {
            writeln!(file, "# ")?;
        }

        // One empty line
        writeln!(file, "")?;

        // #nodes
        let num_nodes = self.nodes.len();
        writeln!(file, "{}", num_nodes)?;
        old_line.clear();
        old_reader.read_line(&mut old_line)?;

        // #edges
        let num_edges = self.num_edges();
        let old_num_edges: usize = old_line.trim().parse().expect("Missing number of edges");
        writeln!(file, "{}", num_edges)?;

        // Print nodes: <ID> <OSMID> <Lat> <Lon> <Height> <Level>
        for i in 0..num_nodes {
            old_line.clear();
            old_reader.read_line(&mut old_line)?;
            writeln!(file, "{} {}", old_line.trim_end(), self.node_at(i).level)?;
        }

        // Print edges: <SrcID> <TrgID> <Weight> <Type> <MaxSpeed> <EdgeIdA> <EdgeIdB>
        let mut prev_source : Option<usize> = None;
        for _ in 0..old_num_edges {
            old_line.clear();
            old_reader.read_line(&mut old_line)?;
            let parts: Vec<&str> = old_line.split_whitespace().collect();
    
            let source: usize = parts[0].parse()?;

            if let Some(prev_source) = prev_source {
                if prev_source != source {
                    for (to, weight, node, edge_id_a, edge_id_b) in self.shortcuts(prev_source) {
                        writeln!(file, "{} {} {} 0 -1 {} {}", prev_source, to, weight, self.calc_outgoing_edge_id(node, edge_id_a), self.calc_incoming_edge_id(node, edge_id_b))?;
                    }
                }
            }

            writeln!(file, "{} -1 -1", old_line.trim_end())?;

            prev_source = Some(source);
        }

        // Print shortcuts of last node
        if let Some(prev_source) = prev_source {
            for (to, weight, node, edge_id_a, edge_id_b) in self.shortcuts(prev_source) {
                writeln!(file, "{} {} {} 0 -1 {} {}", prev_source, to, weight, self.calc_outgoing_edge_id(node, edge_id_a), self.calc_incoming_edge_id(node, edge_id_b))?;
            }
        }

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

