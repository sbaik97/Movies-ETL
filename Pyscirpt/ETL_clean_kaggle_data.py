#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import numpy as np

import re

from sqlalchemy import create_engine
import psycopg2

# from config import db_password
from config import db_password

import time


# In[2]:


# Add the clean movie function that takes in the argument, "movie".
def clean_movie(movie):
# Step 1: Make an empty dict to hold all of the alternative titles.
    movie = dict(movie)  # make a nondestructure copy
    alt_titles_dict = dict()
    # Step 2: Loop through a list of all alternative title keys.
    target_alt_titles = ['Also known as','Arabic','Cantonese','Chinese','French',
            'Hangul','Hebrew','Hepburn','Japanese','Literally',
            'Mandarin','McCune–Reischauer','Original title','Polish',
            'Revised Romanization','Romanized','Russian',
            'Simplified','Traditional','Yiddish']
    for key in target_alt_titles:
# Step 2a: Check if the current key exists in the movie object.
        if key in movie:
# Step 2b: If so, remove the key-value pair and add to the alternative titles dictionary.
            alt_titles_dict[key] = movie[key] # move specific key-value pair into new dictionary
            movie.pop(key)  # remove orginal key-value pairs
# Step 3: After looping through every key, add the alternative titles dict to the movie object.
    if len(alt_titles_dict) > 0:  # make sure it's valid, then add back to movie
        movie['alt_titles'] = alt_titles_dict


    # Inner function for merging column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
            # no return needed
    # Call inner function on every instance in outer function's scope
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')
    # return of outer function
    return movie    


# In[3]:


#  Add the function that takes in three arguments;
# Wikipedia data, Kaggle metadata, and MovieLens rating data (from Kaggle)

def etl_clean(wiki_file, kaggle_file, ratings_file):
    # Read in the kaggle metadata and MovieLens ratings CSV files as Pandas DataFrames.
    kaggle_metadata = pd.read_csv(kaggle_file, low_memory=False)
    ratings = pd.read_csv(ratings_file)   

    # Open and read the Wikipedia data JSON file.
    with open(wiki_file, mode='r') as file:
        wiki_movies_raw = json.load(file)     
    
    #  Write a list comprehension to filter out TV shows.
    wiki_movies = [movie for movie in wiki_movies_raw
    if ('Director' in movie or 'Directed by' in movie)
    and 'imdb_link' in movie
    and 'No. of episodes' not in movie]


    #  Write a list comprehension to iterate through the cleaned wiki movies list
    # and call the clean_movie function on each movie.
    clean_movies = [clean_movie(movie) for movie in wiki_movies]

    #  Read in the cleaned movies list from Step 4 as a DataFrame.
    wiki_movies_df = pd.DataFrame(clean_movies)

    #  Write a try-except block to catch errors while extracting the IMDb ID using a regular expression string and
    #  dropping any imdb_id duplicates. If there is an error, capture and print the exception.
    try:
        wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
        wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
    except:
        print(len(wiki_movies_df))
    

    #  Write a list comprehension to keep the columns that don't have null values from the wiki_movies_df DataFrame.
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns.tolist() if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]


    #  Create a variable that will hold the non-null values from the “Box office” column.
    box_office = wiki_movies_df['Box office'].dropna()
    
    #  Convert the box office data created in Step 8 to string values using the lambda and join functions.
    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)

    #  Write a regular expression to match the six elements of "form_one" of the box office data.
    form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
   
    #  Write a regular expression to match the three elements of "form_two" of the box office data.
    form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'
    box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    #  Add the parse_dollars function.
    
    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan

        # if input is of the form $###.# million
        #We'll use re.match(pattern, string) to see if our string matches a pattern
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):
            # remove dollar sign and " million"
            #use re.sub(pattern, replacement_string, string) to remove dollar signs, spaces, commas, and letters, 
            s = re.sub('\$|\s|[a-zA-Z]','', s)
            # convert to float and multiply by a million
            value = float(s) * 10**6

            # return value
            return value


        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):
            # remove dollar sign and " billion"
            s = re.sub('\$|\s|[a-zA-Z]','', s)  
            # convert to float and multiply by a billion
            value = float(s) * 10**9

            # return value
            return value


        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):
            # remove dollar sign and commas
            s = re.sub('\$|,','', s)
            # convert to float
            value = float(s)

            # return value
            return value

        # otherwise, return NaN
        else:
            return np.nan

        
    #  Clean the box office column in the wiki_movies_df DataFrame.
    wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    wiki_movies_df.drop('Box office', axis=1, inplace=True) 
    
    #  Clean the budget column in the wiki_movies_df DataFrame.
    budget = wiki_movies_df['Budget'].dropna().map(lambda x: ' '.join(x) if type(x) == list else x)
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
    matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
    budget = budget.str.replace(r'\[\d+\]\s*', '')
    wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    wiki_movies_df.drop('Budget', axis=1, inplace=True)
    
    #  Clean the release date column in the wiki_movies_df DataFrame.
    release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'
    release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE) 
    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)
    wiki_movies_df.drop('Release date', axis = 1, inplace = True)
    
    #  Clean the running time column in the wiki_movies_df DataFrame.
    running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)
    wiki_movies_df.drop('Running time', axis=1, inplace=True)
    
    # 2. Clean the Kaggle metadata.
    # keep rows where the adult column is False, and then drop the adult column
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')
    # assign it back to video:
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'
    # Convert data types
    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
    # convert release_date to datetime,
    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])
    # assign it to the timestamp column.
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    
    # 3. Merged the two DataFrames into the movies DataFrame.
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])

    # 4. Drop unnecessary columns from the merged DataFrame.
    # drop the title_wiki, release_date_wiki, Language, and Production company(s) columns
    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)

    # 5. Add in the function to fill in the missing Kaggle data.
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
            , axis=1)
        df.drop(columns=wiki_column, inplace=True)

    # 6. Call the function in Step 5 with the DataFrame and columns as the arguments.
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

    # 7. Filter the movies DataFrame for specific columns.
    movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                           'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                           'genres','original_language','overview','spoken_languages','Country',
                           'production_companies','production_countries','Distributor',
                           'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                          ]]

    # 8. Rename the columns in the movies DataFrame.
    movies_df.rename({'id':'kaggle_id',
                      'title_kaggle':'title',
                      'url':'wikipedia_url',
                      'budget_kaggle':'budget',
                      'release_date_kaggle':'release_date',
                      'Country':'country',
                      'Distributor':'distributor',
                      'Producer(s)':'producers',
                      'Director':'director',
                      'Starring':'starring',
                      'Cinematography':'cinematography',
                      'Editor(s)':'editors',
                      'Writer(s)':'writers',
                      'Composer(s)':'composers',
                      'Based on':'based_on'
                     }, axis='columns', inplace=True)

    # 9. Transform and merge the ratings DataFrame.
    # take the count for each group and rename the "userId" column to "count."
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                     .rename({'userId':'count'}, axis=1)                     .pivot(index='movieId',columns='rating', values='count')
    # rename the columns so they're easier to understand
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]
    # use a left merge
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')
    # make zeros instead of missing values (non).
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)  

    return wiki_movies_df, movies_with_ratings_df, movies_df


# In[4]:


# 10. Create the path to your file directory and variables for the three files.
file_dir = file_dir = 'C://Users/Sungil Baik/Bootcamp/Week-8-ETL/Challenge8/Resources'
# Wikipedia data
wiki_file = f'{file_dir}/wikipedia_movies.json'
# Kaggle metadata
kaggle_file = f'{file_dir}/movies_metadata.csv'
# MovieLens rating data.
ratings_file = f'{file_dir}/ratings.csv'


# In[5]:


# 11. Set the three variables equal to the function created in D1.
wiki_file, kaggle_file, ratings_file = etl_clean(wiki_file, kaggle_file, ratings_file)


# In[6]:


# 12. Set the DataFrames from the return statement equal to the file names in Step 11. 
wiki_movies_df = wiki_file
movies_with_ratings_df = kaggle_file
movies_df = ratings_file


# In[7]:


# 13. Check the wiki_movies_df DataFrame. 
wiki_movies_df.head()


# In[8]:


# 14. Check the movies_with_ratings_df DataFrame.
movies_with_ratings_df.head()


# In[9]:


# 15. Check the movies_df DataFrame. 
movies_df.head()

