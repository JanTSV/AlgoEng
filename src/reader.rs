use std::fs::File;
use std::io::{BufRead, BufReader};
use std::error::Error;

use crate::graph::{Node, Edge, OffsetArray};

pub fn parse_graph(filename: &str) -> Result<OffsetArray, Box<dyn Error>> {
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

     // Parse nodes
     let mut nodes: Vec<Node> = vec![];
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
 
         nodes.push(Node::new(osm_id, lat, lon, height, level));
     }
 
     // Build edge lists
     let mut edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes];
     let mut reverse_edges: Vec<Vec<Edge>> = vec![Vec::new(); num_nodes];
 
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
 
         edges[source].push(Edge::new(target, weight, typ, max_speed, edge_id_a, edge_id_b));
         reverse_edges[target].push(Edge::new(source, weight, typ, max_speed, edge_id_b, edge_id_a));
     }
 

    Ok(OffsetArray::build_from(nodes, edges, reverse_edges))
}

pub fn parse_queries(filename: &str) -> Result<Vec<(usize, usize)>, Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    Ok(reader
        .lines()
        .map_while(Result::ok)
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                return None;
            }

            let source: usize = parts[0].parse().ok()?;
            let target: usize = parts[1].parse().ok()?;

            Some((source, target))
        })
        .collect())
}

#[cfg(test)]
mod reader_tests {
    use super::{parse_graph, parse_queries};

    #[test]
    fn test_parse_toy() {
        let graph = parse_graph("inputs/toy.fmi").unwrap();
        assert_eq!(graph.nodes.len(), 5);
        assert_eq!(graph.edges.len(), 9);
        assert_eq!(graph.offsets.len(), 6);
        assert_eq!(graph.reverse_offsets.len(), 6);
    }
    
    #[test]
    fn test_parse_mv() {
        let graph = parse_graph("inputs/MV.fmi").unwrap();
        assert_eq!(graph.nodes.len(), 644199);
        assert_eq!(graph.edges.len(), 1305996);
        assert_eq!(graph.offsets.len(), 644200);
        assert_eq!(graph.reverse_offsets.len(), 644200);
    }

    #[test]
    fn test_parse_querries() {
        let queries = parse_queries("inputs/queries.txt").unwrap();

        const EXPECTED: [(usize, usize); 40] = [
            (214733, 429466),
            (214733, 429467),
            (214733, 429468),
            (214733, 429469),
            (214733, 429470),
            (214733, 429471),
            (214733, 429472),
            (214733, 429473),
            (214733, 429474),
            (214733, 429475),
            (214734, 429466),
            (214734, 429467),
            (214734, 429468),
            (214734, 429469),
            (214734, 429470),
            (214734, 429471),
            (214734, 429472),
            (214734, 429473),
            (214734, 429474),
            (214734, 429475),
            (214735, 429466),
            (214735, 429467),
            (214735, 429468),
            (214735, 429469),
            (214735, 429470),
            (214735, 429471),
            (214735, 429472),
            (214735, 429473),
            (214735, 429474),
            (214735, 429475),
            (214736, 429466),
            (214736, 429467),
            (214736, 429468),
            (214736, 429469),
            (214736, 429470),
            (214736, 429471),
            (214736, 429472),
            (214736, 429473),
            (214736, 429474),
            (214736, 429475)];

        assert_eq!(EXPECTED.len(), queries.len());

        for (i, q) in queries.iter().enumerate() {
            assert_eq!(EXPECTED[i], *q);
        }
    }
}