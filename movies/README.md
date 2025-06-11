# Solution

This is the solution for Problem 2.

To get started, simply run
```bash
make
```

This should build an executable (`movies`) and copy it to your working directory.

The path to the `movies.txt` file is inputted with a command line parameter:
```
Usage: ./movies <movies.txt>
```

Running
```bash
./movies movies.txt
```

should output something along these lines
```
Parsing movies.txt...
Parsed in 96.84ms
Creating hashmap...
Created hashmap in 1.06s
Creating search tree...
Created search tree in 2.63s
Server running at http://localhost:8080/
```

Now navigate to the given link using a browser of your choice. 

You should see a `Query` input field:
- ` ` (whitespace) is the delimiter of the query words. All words must occur in the movie description.
- The query is not case-sensitive.

Below that, there should be a dropdown menu to choose the 3 variants for the search mode (`naive`, `hashmap`, and `tree`). 
Select one of the options and click on submit. This runs the query and displays a table of the found movies as well as the execution time of the query.
The movie list outputted using `hashmap` and `tree` is sorted by the `idf * tf` technique as discussed in the script.

# Comparison

Unlike the first solution, I did not provide an executable unittest.
But from playing around it seems like the search tree is usually faster than the hashmap.
The naive solution is obviously very slow.