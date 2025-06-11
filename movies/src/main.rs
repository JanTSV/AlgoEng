use std::{env, time::Instant};
use std::fs::File;
use std::io::{BufRead, BufReader};
use warp::Filter;
use std::sync::Arc;
use std::collections::{HashMap, BTreeMap};

#[derive(Debug)]
struct Movie {
    title: String,
    description: String
}

impl Movie {
    fn new(title: String, description: String) -> Self {
        Movie { title, description }
    }
}

fn iter_words(s: &String) -> impl Iterator<Item = String> + '_ {
    s
        .split_whitespace()
        .map(|word| {
            word
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'' || *c == '-')
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
}

fn parse(filename: &str) -> Vec<Movie> {
    let file = File::open(filename).expect("Could not open {filename}");
    let reader = BufReader::new(file);

    reader
        .lines()
        .map_while(Result::ok)
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 2 {
                return None;
            }
            Some(Movie::new(parts[0].to_string(), parts[1].to_string()))
        })
        .collect()
}

#[derive(Clone)]
struct MovieRank {
    // TODO: Compression
    id: usize,
    relevance: f32,
}

impl MovieRank {
    fn new(id: usize) -> Self {
        MovieRank { id, relevance: 1.0 }
    }
}

fn naive<'a>(movies: &'a [Movie], query: &[String]) -> Vec<MovieRank> {
    let mut found = Vec::new();
    let mut last = 0;

    'outer: for (i, movie) in movies.iter().enumerate() {
        // let offset = i - last;
        for q in query {
            if !iter_words(&movie.description).any(|w| w == *q) {
                // If one word doesn't match, skip this movie
                continue 'outer;
            }
        }

        // All query words matched
        found.push(MovieRank::new(i));
        last = i;
    }

    found
}

fn create_inverted_index_hash_map(movies: &[Movie]) -> (HashMap<String, Vec<MovieRank>>, Vec<f32>) {
    let mut hashmap: HashMap<String, Vec<MovieRank>> = HashMap::new();
    let mut max_tf = vec![1.0f32; movies.len()];

    // let mut last = 0;
    for (i, movie) in movies.iter().enumerate() {
        // let offset = i - last;
        for word in iter_words(&movie.description) {
            match hashmap.get_mut(&word) {
                Some(x) =>  {
                    match x.last_mut() {
                        Some(y) if y.id == i => {
                            // First relevance just counts the number of occurences
                            y.relevance += 1.0;
                            max_tf[i] += 1.0;
                        },
                        _ => {
                            x.push(MovieRank::new(i));
                            // last = i;

                        }
                    }
                },
                None => {
                    let mut v = Vec::new();
                    v.push(MovieRank::new(i));
                    hashmap.insert(word, v);
                    // last = i;
                }
            }
        }
    }

    (hashmap, max_tf)
}

fn create_inverted_index_tree(movies: &[Movie]) -> BTreeMap<String, Vec<MovieRank>> {
    let mut tree: BTreeMap<String, Vec<MovieRank>> = BTreeMap::new();

    // let mut last = 0;
    for (i, movie) in movies.iter().enumerate() {
        for word in iter_words(&movie.description) {
            match tree.get_mut(&word) {
                Some(x) =>  {
                    match x.last_mut() {
                        Some(y) if y.id == i => {
                            // First relevance just counts the number of occurences
                            y.relevance += 1.0;
                        },
                        _ => {
                            x.push(MovieRank::new(i));
                        }
                    }
                },
                None => {
                    let mut v = Vec::new();
                    v.push(MovieRank::new(i));
                    tree.insert(word, v);
                }
            }
        }
    }

    tree
}

fn binary_search<'a>(v: &'a [MovieRank], s: &MovieRank, mut left: usize, mut right: usize) -> Option<&'a MovieRank> {
    while left < right {
        let mid = left + (right - left) / 2;
        if v[mid].id == s.id {
            return Some(&v[mid]);
        } else if v[mid].id < s.id {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    None
}

fn galloping_search<'a>(v: &'a [MovieRank], s: &MovieRank) -> Option<&'a MovieRank> {
    if v.is_empty() {
        return None;
    }

    let mut bound = 1;

    while bound < v.len() && v[bound].id < s.id {
        bound *= 2;
    }

    binary_search(v, s, bound / 2, (bound + 1).min(v.len()))
}

fn intersect_galopping_search(a: &[MovieRank], b: &[MovieRank], n: usize, m: usize, max_tf: &[f32]) -> Vec<MovieRank> {
    b.iter().filter_map(|bb| if let Some(aa) = galloping_search(a, bb) { Some(MovieRank{ id: aa.id, relevance: aa.relevance + calc_relevance(n, m, bb.relevance, max_tf[aa.id])}) } else { None }).collect()
}

fn calc_relevance(n: usize, m: usize, occurences: f32, max_tf: f32) -> f32 {
    let idf = (n as f32 / m as f32).ln();
    let tf = occurences / max_tf;
    idf * tf
}

fn query_hashmap(hashmap: &HashMap<String, Vec<MovieRank>>, query: &[String], max_tf: &[f32], n: usize) -> Vec<MovieRank> {
    let mut found: Vec<MovieRank> = Vec::new();
    let mut first = true;

    for q in query {
        match hashmap.get(q) {
            Some(q_found) => {
                match first {
                    false => found = intersect_galopping_search(&found, &q_found, n, q_found.len(), max_tf),
                    true => {
                        found.extend(q_found.iter().map(|x| {
                            let relevance = calc_relevance(n, q_found.len(), x.relevance, max_tf[x.id]);
                            MovieRank { id: x.id, relevance }
                        }));
                        first = false;
                    }
                }
            },
            None => continue
        } 
    }

    found
}

fn query_tree(tree: &BTreeMap<String, Vec<MovieRank>>, query: &[String], max_tf: &[f32], n: usize) -> Vec<MovieRank> {
    let mut found: Vec<MovieRank> = Vec::new();
    let mut first = true;

    for q in query {
        match tree.get(q) {
            Some(q_found) => {
                match first {
                    false => found = intersect_galopping_search(&found, &q_found, n, q_found.len(), max_tf),
                    true => {
                        found.extend(q_found.iter().map(|x| {
                            let relevance = calc_relevance(n, q_found.len(), x.relevance, max_tf[x.id]);
                            MovieRank { id: x.id, relevance }
                        }));
                        first = false;
                    }
                }
            },
            None => continue
        } 
    }

    found
}

fn query<'a>(data: FormData, movies: &'a [Movie], hashmap: &HashMap<String, Vec<MovieRank>>, tree: &BTreeMap<String, Vec<MovieRank>>, max_tf: &Vec<f32>) -> String {
    let query: Vec<String> = data.query.split_whitespace().map(|q| q.to_string().to_lowercase()).collect();

    let start = Instant::now();
    let found = match data.method.as_str() {
        "naive" => Some(naive(movies, &query)),
        "hashmap" => Some(query_hashmap(hashmap, &query, &max_tf, movies.len())),
        "tree" => Some(query_tree(tree, &query, &max_tf, movies.len())),
        _ => None
    };
    let duration = start.elapsed();

    if let Some(mut found) = found {
        // Sort found by relevance
        found.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());

        let mut table = String::from("<table border=1><tr><th>Title</th><th>Description</th></tr>");
        // let mut i = 0;
        for rank in found {
            // i += rank.offset;
            let i = rank.id;
            table.push_str(format!("<tr><td>{}</td><td>{}</td></tr>", movies[i].title, movies[i].description).as_str())
        }

        table.push_str("</table>");
        
        format!(
            r#"
            <html>
            <head><title>Results</title></head>
            <body>
                <h2>Results for: {}</h2>
                {}: {:.2?}
                <hr>
                {}
                <br>
                <a href="/">Back</a>
            </body>
            </html>
            "#,
            data.query, data.method, duration, table
        )
    } else {
        String::from(r#"
            <html>
            <head><title>Results</title></head>
            <body>
                <h2>Invalid parameters</h2>
                <br>
                <a href="/">Back</a>
            </body>
            </html>
            "#)
    }

}

#[tokio::main]
async fn main() {
    // Read console args
   let args: Vec<String> = env::args().collect();
    const PORT: u16 = 8080;

    if args.len() != 2 {
        println!("Usage: ./movies <movies.txt>");
        return;
    }

    // Parse movie file
    println!("Parsing {}...", args[1]);
    let s = Instant::now();
    let movies = Arc::new(parse(&args[1]));
    //let movies = Arc::new(parse("movies.txt"));
    println!("Parsed in {:.2?}", s.elapsed());

    // Create inverted index with hashmap
    println!("Creating hashmap...");
    let s = Instant::now();
    let (hashmap, max_tf) = create_inverted_index_hash_map(&movies);
    let hashmap = Arc::new(hashmap);
    let max_tf = Arc::new(max_tf);
    println!("Created hashmap in {:.2?}", s.elapsed());

    // Create inverted index with search tree
    println!("Creating search tree...");
    let s = Instant::now();
    let tree = Arc::new(create_inverted_index_tree(&movies));
    println!("Created search tree in {:.2?}", s.elapsed());

    // Simple HTML server
    let form = warp::path::end()
        .map(|| warp::reply::html(FORM_HTML));

    // Handle query
    let movies_filter = warp::any().map({
        let movies = Arc::clone(&movies);
        move || Arc::clone(&movies)
    });

    let hashmap_filter = warp::any().map({
        let hashmap = Arc::clone(&hashmap);
        move || Arc::clone(&hashmap)
    });

    let tree_filter = warp::any().map({
        let tree = Arc::clone(&tree);
        move || Arc::clone(&tree)
    });

    let max_tf_filter = warp::any().map({
        let max_tf = Arc::clone(&max_tf);
        move || Arc::clone(&max_tf)
    });

    let submit = warp::path("submit")
        .and(warp::query::<FormData>())
        .and(movies_filter)
        .and(hashmap_filter)
        .and(tree_filter)
        .and(max_tf_filter)
        .map(|data: FormData, movies: Arc<Vec<Movie>>, hashmap: Arc<HashMap<String, Vec<MovieRank>>>, tree: Arc<BTreeMap<String, Vec<MovieRank>>>, max_tf: Arc<Vec<f32>>| {
            warp::reply::html(query(data, &movies, &hashmap, &tree, &max_tf))
        });

    let routes = form.or(submit);
    
    // Host server
    println!("Server running at http://localhost:{PORT}/");
    warp::serve(routes).run(([127, 0, 0, 1], PORT)).await;
}

#[derive(Debug, serde::Deserialize)]
struct FormData {
    query: String,
    method: String,
}

const FORM_HTML: &str = r#"
<!DOCTYPE html>
<html>
<head><title>Form</title></head>
<body>
    <h2>MOVIE QUERY</h2>
    <form action="/submit" method="get">
        <label>Query:</label><br>
        <input type="text" name="query"><br><br>

        <label>Option:</label><br>
        <select name="method">
            <option value="naive">naive</option>
            <option value="hashmap">hashmap</option>
            <option value="tree">tree</option>
        </select><br><br>

        <input type="submit" value="Submit">
    </form>
</body>
</html>
"#;