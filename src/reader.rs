use std::fs::File;
use std::io::{BufRead, BufReader};
use std::error::Error;

use crate::graph::NodeId;

pub fn parse_queries(filename: &str) -> Result<Vec<(NodeId, NodeId)>, Box<dyn Error>> {
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

            let source: NodeId = parts[0].parse().ok()?;
            let target: NodeId = parts[1].parse().ok()?;

            Some((source, target))
        })
        .collect())
}

#[cfg(test)]
mod reader_tests {
    use super::*;

    #[test]
    fn test_parse_querries() {
        let queries = parse_queries("inputs/querries.txt").unwrap();

        const EXPECTED: [(NodeId, NodeId); 40] = [
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