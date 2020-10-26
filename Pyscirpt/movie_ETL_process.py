#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import numpy as np


# In[2]:


file_dir = 'C://Users/Sungil Baik/Bootcamp/Week-8-ETL/archive/'


# In[3]:


f'{file_dir}wikipedia-movies.json'


# In[4]:


with open(f'{file_dir}wikipedia-movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)


# In[5]:


# First 5 records
wiki_movies_raw[:5]


# In[6]:


# Last 5 records
wiki_movies_raw[-5:]


# In[7]:


kaggle_metadata = pd.read_csv(f'{file_dir}/movies_metadata.csv', low_memory=False)
ratings = pd.read_csv(f'{file_dir}ratings.csv')


# In[8]:


ratings.sample(n=5) 


# In[9]:


wiki_movies_df = pd.DataFrame(wiki_movies_raw)


# In[10]:


wiki_movies_df.head()


# In[11]:


wiki_movies_df.columns.tolist()


# In[12]:


wiki_movies = [movie for movie in wiki_movies_raw
if ('Director' in movie or 'Directed by' in movie)
and 'imdb_link' in movie
and 'No. of episodes' not in movie]
len(wiki_movies)


# ### Function test

# In[13]:


my_list = [1,2,3]
def append_four(x):
    x.append(4)
append_four(my_list)
print(my_list)


# In[14]:


# Lambda Functions
square = lambda x: x * x
square(5)


# In[ ]:





# ### Create a Function to Clean the Data

# In[15]:


# look at the Arabic language.
wiki_movies_df[wiki_movies_df['Arabic'].notnull()]['url']


# In[16]:


sorted(wiki_movies_df.columns.tolist())


# In[17]:


wiki_movies_df[wiki_movies_df['Polish'].notnull()]['url']


# In[18]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    # combine alternate titles into one list
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune-Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

    # merge column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
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

    return movie


# In[19]:


clean_movies = [clean_movie(movie) for movie in wiki_movies]
wiki_movies_df = pd.DataFrame(clean_movies)
sorted(wiki_movies_df.columns.tolist())


# In[20]:


wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))
wiki_movies_df.head()


# In[21]:


# Count Null Columns
[[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]


# In[22]:


#make a list of columns that have less than 90% null values and use those to trim down our dataset
[column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]


# In[23]:


#select the columns that we want to keep
wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]
wiki_movies_df


# In[24]:


#display the data type for each column.
wiki_movies_df.dtypes


# In[25]:


# make a data series that drops missing values with the following:
box_office = wiki_movies_df['Box office'].dropna()


# In[26]:


#By using the apply() method, we can see which values are not strings.
def is_not_a_string(x):
    return type(x) != str


# In[27]:


box_office[box_office.map(is_not_a_string)]


# In[28]:


# Instead of creating a new function with a block of code and the def keyword, 
# we can create an anonymous lambda function right inside the map() call
box_office[box_office.map(lambda x: type(x) != str)]


# In[29]:


#use a simple space as our joining character 
#and apply the join() function only when our data points are lists. 
box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)


# ### Parse the Box Office Data

# In[30]:


import re


# In[31]:


# Create a variable form_one 
# and set it equal to the finished regular expression string
# flags=re.IGNORECASE is used to ignore whether letters are uppercase or lowercase
form_one = r'\$\d+\.?\d*\s*[mb]illion'
box_office.str.contains(form_one, flags=re.IGNORECASE).sum()


# In[32]:


#To be safe, we should see if any box offce values are described by both.
# Create another variable form_two 
#and set it equal to the fnished regular expression string.
form_two = r'\$\d{1,3}(?:,\d{3})+'
box_office.str.contains(form_two, flags=re.IGNORECASE).sum()


# In[33]:


# To make our code easier to understand, we'll create two Boolean Series
# called matches_form_one and matches_form_two , and then select the box
# offce values that don't match either.
matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)
# Pandas has element-wise logical operators:~ (negation operator), &,|
box_office[~matches_form_one & ~matches_form_two]


# In[34]:


#fix our pattern matches to capture more values by addressing few issues
#1. Some values have spaces in between the dollar sign and the number.
form_one = r'\$\s*\d+\.?\d*\s*[mb]illion'
form_two = r'\$\s*\d{1,3}(?:,\d{3})+'
#2. Some values use a period as a thousands separator, not a comma.
#(ex. 1.234 billion --> $123.456.789.)
form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'
#3. Some values are given as a range.
box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
#4. "Million" is sometimes misspelled as "millon."
form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'


# In[35]:


#Extract and Convert the Box Office value
# captures data when it matches either form_one or form_two using an f-string
box_office.str.extract(f'({form_one}|{form_two})')


# In[36]:


# parse_dollars will take in a string and return a foating-point number
# function to turn the extracted values into a numeric value.
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


# In[37]:


# parse the box office values to numeric values
wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
#the output
wiki_movies_df['box_office']


# In[38]:


#no longer need the Box Office column, so we'll just drop it:
wiki_movies_df.drop('Box office', axis=1, inplace=True)  


# ### Parse Budget Data

# In[39]:


#Create a budget variable with the following code:
budget = wiki_movies_df['Budget'].dropna()


# In[40]:


#Convert any lists to strings:
budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)


# In[41]:


# Remove any values between a dollar sign and a hyphen 
budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[42]:


#Use the same pattern matches that you created to parse the box office data
matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
budget[~matches_form_one & ~matches_form_two]


# In[43]:


#Remove the citation references with the following:
budget = budget.str.replace(r'\[\d+\]\s*', '')
budget[~matches_form_one & ~matches_form_two]


# In[44]:


# parse the box office values, changing "box_office" to "budget":
wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
# drop the original Budget column.
wiki_movies_df.drop('Budget', axis=1, inplace=True)


# ### Parse Release Date

# In[45]:


# First, make a variable that holds the non-null values of Release date 
# in the DataFrame, converting lists to strings:
release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[46]:


#Full month name, one- to two-digit day, four-digit year (i.e., January 1, 2000)
date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
#Four-digit year, two-digit month, two-digit day, with any separator (i.e., 2000-01-01)
date_form_two = r'\d{4}.[01]\d.[123]\d'
#Full month name, four-digit year (i.e., January 2000)
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
#Four-digit year
date_form_four = r'\d{4}'


# In[47]:


# Extract the dates with:
release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)   


# In[48]:


#The date formats we've targeted are among those that 
#the to_datetime() function can recognize,
wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)
wiki_movies_df['release_date']


# ### Parse Running Time

# In[49]:


# make a variable that holds the non-null values of Release date 
# in the DataFrame, converting lists to strings
running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[50]:


# see how many running times look exactly like that by using string boundaries.
running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE).sum()


# In[51]:


#  get a sense of what the other 366 entries look like.
running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]


# In[52]:


# make this more general by only marking the beginning of the string, 
# and accepting other abbreviations of "minutes"
running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()


# In[53]:


#The remaining 17 follow:
running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]


# In[54]:


#match all of the hour + minute patterns with one regular expression pattern
#Extract digits, and allow for both possible patterns.
running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')


# In[55]:


# DataFrame is all strings, need to convert them to numeric values.
#  use the to_numeric() method and set the errors argument to 'coerce' which turn the erros to Not a Number (NaN)
running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)


# In[56]:


# Aapply a function that will convert the hour capture groups 
# and minute capture groups to minutes if the pure minutes capture group is zero
wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)


# In[57]:


# rop Running time from the dataset with the following code:
wiki_movies_df.drop('Running time', axis=1, inplace=True)


# In[58]:


# output
wiki_movies_df['running_time']


# In[59]:


wiki_movies_df


# ### Clean the Kaggle Data

# In[60]:


#check all of the columns of Kaggle Data came in as the correct data types.
kaggle_metadata.dtypes


# In[61]:


# Before we convert the "adult" and "video" columns, we want to check that
# all the values are either True or False .
kaggle_metadata['adult'].value_counts()


# In[62]:


#find and remove the adult movie
kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]


# In[63]:


# keep rows where the adult column is False, and then drop the adult column
kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')


# In[64]:


#look at the values of the video column
kaggle_metadata['video'].value_counts()


# In[65]:


# convert the video column using the following code:
kaggle_metadata['video'] == 'True'


# In[66]:


# assign it back to video:
kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'


# In[67]:


# Use the to_numeric() method from Pandas to convert Data Types
# If there's any data that can't be converted to numbers, 
# make sure the errors= argument is set to 'raise'
kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')


# In[68]:


# convert release_date to datetime, 
# Pandas has a built-in function, to_datetime().
kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])


# ### Reasonability Checks on Ratings Data, 

# In[69]:


# take a look at the ratings data, use the info() method, and set the null_counts option to True
ratings.info(null_counts=True)


# In[70]:


# specify in to_datetime() that the origin is 'unix' and the time unit is seconds.
pd.to_datetime(ratings['timestamp'], unit='s')


# In[71]:


# assign it to the timestamp column.
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')


# In[72]:


# statistics of the actual ratings using the barchart and describe() method 
# bar chart needs how often a data point shows up in the data.
pd.options.display.float_format = '{:20,.2f}'.format
ratings['rating'].plot(kind='hist')
ratings['rating'].describe()


# In[73]:


#final check all of the columns came in as the correct data types.
kaggle_metadata.dtypes


# In[ ]:





# # Merge Wikipedia and Kaggle Metadata

# In[74]:


# Print out a list of the columns so we can identify which ones are redundant.
movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])


# In[75]:


movies_df.head(3)


# In[76]:


# making a list of competing columns
movies_df[['title_wiki','title_kaggle']].head(3)


# ### Title

# In[77]:


#Look at the rows where the titles don't match.
movies_df[movies_df['title_wiki'] != movies_df['title_kaggle']][['title_wiki','title_kaggle']].head(4)


# In[78]:


# Show any rows where title_kaggle is empty
movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]


# In[79]:


# Running time _ fill in missing values with zero and make the scatter plot for running time
movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')


# ### Budget

# In[80]:


# ake another scatter plot to compare the values
movies_df.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')


# ### Box Office 

# In[81]:


# both are numeric, make another scatter plot
movies_df.fillna(0).plot(x='box_office', y='revenue', kind='scatter')


# In[82]:


#look at the scatter plot for everything less than $1 billion in box_office
movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')


# ### Release Date

# In[83]:


# use the regular line plot (which can plot date data), 
# and change the style to only put dots by adding style='.' to the plot() method:
movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')


# In[84]:


# any movie whose release date according to Wikipedia is after 1996, 
# but whose release date according to Kaggle is before 1965. 
movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]


# In[85]:


# drop that row from our DataFrame. 
# We'll get the index of that row with the following:
movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index
#drop that row like this
movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)


# In[86]:


# see if there are any null values:
movies_df[movies_df['release_date_wiki'].isnull()]


# ### Language

# In[87]:


# compare the value counts of each. However, consider the following code:
movies_df['Language'].value_counts()


# In[88]:


# convert the lists in Language to tuples 
# so that the value_counts() method will work
movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)


# In[89]:


# For the Kaggle data, there are no lists, we can run fuction, value_counts()
movies_df['original_language'].value_counts(dropna=False)


# ### Production Companies

# In[90]:


movies_df.columns.tolist()


# In[91]:


movies_df[['Production company(s)','production_companies']]


# In[ ]:





# ### Put It All Together

# In[92]:


# drop the title_wiki, release_date_wiki, Language, and Production company(s) columns
movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)


# In[93]:


# make a function that fills in missing data for a column pair 
# and then drops the redundant column.
def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)


# In[94]:


#run the function for the three column pairs that we decided to fill in zeros
fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df.head(3)


# In[95]:


# check that there aren't any columns with only one value, 
# since that doesn't really provide any information
#need to convert lists to tuples for value_counts() to work.
for col in movies_df.columns:
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    if num_values == 1:
        print(col)


# In[96]:


movies_df['video'].value_counts(dropna=False)


# In[97]:


movies_df.dtypes


# In[98]:


movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]


# In[99]:


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


# In[100]:


movies_df.head(3)


# # Transform and Merge Rating Data

# In[101]:


# See a ratings
ratings.sample(n=5)


# In[102]:


# use a groupby on the "movieId" and "rating" columns and take the count for each group.
# rename the "userId" column to "count."
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)


# In[103]:


#pivot this data so that movieId is the index, the columns will be all the rating values,
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)                 .pivot(index='movieId',columns='rating', values='count')
rating_counts.head(3)


# In[104]:


# rename the columns so they're easier to understand. We'll prepend rating_ to each column with a list comprehension:
rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]
rating_counts.head(3)


# In[105]:


# use a left merge
movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')
movies_with_ratings_df.head(3)


# In[106]:


# make zeros instead of missing values (non).
movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)
movies_with_ratings_df.head(3)


# In[ ]:





# ### Connect Pandas and SQL, and Print Elapsed Time

# In[107]:


# Import sqlalchemy Modules and  config password file
from sqlalchemy import create_engine
import psycopg2
from config import db_password
import time


# In[108]:


# LOAD MOVIES_DF TO Postgres
db_string = f'postgres://postgres:{db_password}@127.0.0.1:57393/movie_data'


# In[109]:


# use sqlalchemy.create_engine to prepare parameter of pd.to_sql()
engine = create_engine(db_string)


# In[ ]:





# In[ ]:


# create a variable for the number of rows imported
rows_imported = 0
# get the start_time from time.time()
start_time = time.time()
for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):

    # print out the range of rows that are being imported
    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')

    data.to_sql(name='ratings', con=engine, if_exists='append')

    # increment the number of rows imported by the size of 'data'
    rows_imported += len(data)

    # add elapsed time to final print out
    print(f'Done. {time.time() - start_time} total seconds elapsed')


# In[ ]:


#save the movies_df DataFrame to a SQL table,
movies_df.to_sql(name='movies', con=engine)


# 
