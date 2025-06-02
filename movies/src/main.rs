use std::{env, time::Instant};
use std::fs::File;
use std::io::{BufRead, BufReader};
use warp::Filter;
use std::sync::Arc;
use std::collections::HashMap;

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
    rank: u16
}

impl MovieRank {
    fn new(id: usize) -> Self {
        MovieRank { id, rank: 1 }
    }
}

impl PartialEq for MovieRank {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for MovieRank {}

impl PartialOrd for MovieRank {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.id.cmp(&other.id))
    }
}

impl Ord for MovieRank {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
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

fn create_inverted_index_hash_map(movies: &[Movie]) -> HashMap<String, Vec<MovieRank>> {
    let mut hashmap: HashMap<String, Vec<MovieRank>> = HashMap::new();

    // let mut last = 0;
    for (i, movie) in movies.iter().enumerate() {
        // let offset = i - last;
        for word in iter_words(&movie.description) {
            match hashmap.get_mut(&word) {
                Some(x) =>  {
                    match x.last_mut() {
                        Some(y) if y.id == i => y.rank += 1,
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

    hashmap
}

fn binary_search<Item: PartialOrd + PartialEq>(v: &[Item], s: &Item, mut left: usize, mut right: usize) -> bool {
    while left < right {
        let mid = left + (right - left) / 2;
        if v[mid] == *s {
            return true;
        } else if v[mid] < *s {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    false
}

fn intersect_galopping_search<Item: PartialOrd + PartialEq + Clone>(a: &[Item], b: &[Item]) -> Vec<Item> {
    b.iter().filter_map(|be| if galloping_search(a, be) { Some(be.clone()) } else { None }).collect()
}

fn galloping_search<Item: PartialOrd + PartialEq>(v: &[Item], s: &Item) -> bool {
    if v.is_empty() {
        return false;
    }

    let mut bound = 1;

    while bound < v.len() && v[bound] < *s {
        bound *= 2;
    }

    binary_search(v, s, bound / 2, (bound + 1).min(v.len()))
}

fn query_hashmap(hashmap: &HashMap<String, Vec<MovieRank>>, query: &[String]) -> Vec<MovieRank> {
    let mut found: Vec<MovieRank> = Vec::new();
    let mut first = true;

    for q in query {
        match hashmap.get(q) {
            Some(q_found) => {
                match first {
                    false => found = intersect_galopping_search(&found, &q_found),
                    true => {
                        found.extend(q_found.iter().cloned());
                        first = false;
                    }
                }
            },
            None => continue
        } 
    }

    found
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
    println!("Parsed in {:.2?}", s.elapsed());

    // TODO: Create inverted index with hashmap
    println!("Creating hashmap...");
    let s = Instant::now();
    let hashmap = Arc::new(create_inverted_index_hash_map(&movies));
    println!("Created hashmap in {:.2?}", s.elapsed());

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

    let submit = warp::path("submit")
        .and(warp::query::<FormData>())
        .and(movies_filter)
        .and(hashmap_filter)
        .map(|data: FormData, movies: Arc<Vec<Movie>>, hashmap: Arc<HashMap<String, Vec<MovieRank>>>| {
            warp::reply::html(query(data, &movies, &hashmap))
        });

    let routes = form.or(submit);
    
    // Host server
    println!("Server running at http://localhost:{PORT}/");
    warp::serve(routes).run(([127, 0, 0, 1], PORT)).await;
}

fn query<'a>(data: FormData, movies: &'a [Movie], hashmap: &HashMap<String, Vec<MovieRank>>) -> String {
    let query: Vec<String> = data.query.split_whitespace().map(|q| q.to_string().to_lowercase()).collect();

    let start = Instant::now();
    let found = match data.method.as_str() {
        "naive" => Some(naive(movies, &query)),
        "hashmap" => Some(query_hashmap(hashmap, &query)),
        _ => None
    };
    let duration = start.elapsed();

    if let Some(found) = found {
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
            <option value="option2">option2</option>
        </select><br><br>

        <input type="submit" value="Submit">
    </form>
</body>
</html>
"#;