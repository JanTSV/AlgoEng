use std::{env, time::Instant};
use std::fs::File;
use std::io::{BufRead, BufReader};
use warp::Filter;
use std::sync::Arc;

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

fn naive<'a>(movies: &'a [Movie], query: &[String]) -> Vec<&'a Movie> {
    let s = Instant::now();
    let mut found = Vec::new();

    'outer: for movie in movies {
        let words: Vec<&str> = movie.description.split_whitespace().collect();

        for q in query {
            if !words.iter().any(|&w| w == q) {
                continue 'outer; // If one word doesn't match, skip this movie
            }
        }

        // All query words matched
        found.push(movie);
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

    // Simple HTML server
    let form = warp::path::end()
        .map(|| warp::reply::html(FORM_HTML));

    // Handle query
    let movies_filter = warp::any().map({
        let movies = Arc::clone(&movies);
        move || Arc::clone(&movies)
    });

    let submit = warp::path("submit")
        .and(warp::query::<FormData>())
        .and(movies_filter)
        .map(|data: FormData, movies: Arc<Vec<Movie>>| {
            warp::reply::html(query(&movies, data))
        });

    let routes = form.or(submit);
    
    // Host server
    println!("Server running at http://localhost:{PORT}/");
    warp::serve(routes).run(([127, 0, 0, 1], PORT)).await;
}

fn query<'a>(movies: &'a [Movie], data: FormData) -> String {
    let query: Vec<String> = data.query.split_whitespace().map(|q| q.to_string()).collect();

    let start = Instant::now();
    let found = match data.method.as_str() {
        "naive" => Some(naive(movies, &query)),
        _ => None
    };
    let duration = start.elapsed();

    if let Some(found) = found {
        let mut table = String::from("<table border=1><tr><th>Title</th><th>Description</th></tr>");

        for movie in found {
            table.push_str(format!("<tr><td>{}</td><td>{}</td></tr>", movie.title, movie.description).as_str())
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
            <option value="option1">option1</option>
            <option value="option2">option2</option>
        </select><br><br>

        <input type="submit" value="Submit">
    </form>
</body>
</html>
"#;