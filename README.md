# Movies-ETL
Extract, Transform, and Load the movie files

### Background
Analysis is impossible without access to good data, so creating data pipelines is the first step before any analysis can be performed. Herein, we extract, transform and load (ETL) from Wikipedia(JSON format), Kaggle(csv file) MovieLens_Ratings (csv file) into PostgreSQL,to create data pipelines using python and pandas.

### Resources:

- an web-scraped JSON file of over 5,000 movies from 1990 to 2019, from wikipedia [wikipedia.movies.json](Resources/wikipedia.movies.json)

- a csv file from Kaggle [movies_metadata.csv](Resources/movies_metadata.csv)

- a csv file from MovieLens with movie rating information [movies_metadata.csv](Resources/ratings.csv)



### Outputs:


- ETL Jupyter NoteBooks [movies_ETL.ipynb](/movies_ETL.ipynb)


# Challenge

## Goals:

- Create an automated ETL pipeline.
 
- Extract data from multiple sources.

- Clean and transform the data automatically using Pandas and regular expressions.

- Load new data into existing tables in PostgreSQL

### Challenge Outputs:

- Challenge automated Pipeline Python script[challenge.py](/challenge.py)

- Challenge automated Pipeline Jupyter NoteBooks [challenge.ipynb](/challenge.ipynb)
