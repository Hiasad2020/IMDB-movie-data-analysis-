#!/usr/bin/env python
# coding: utf-8

# In[4]:


#IMDB Movie Data Analysis

#We have the data for the 100 top-rated movies from the past decade along with various pieces of information about the movie, 
#its actors, and the voters who have rated these movies online. In this project, 
#We will try to find some interesting insights into these movies and their voters, using Python.


# In[3]:


# Filtering out the warnings
import warnings
warnings.filterwarnings('ignore')


# In[5]:


# Importing the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


###  Task 1: Reading the data

#### Subtask 1.1: Read the Movies Data.

##Read the movies data file provided and store it in a dataframe `movies`.


# In[17]:


# Read the csv file using 'read_csv'
movies = pd.read_csv('Movie_Data.csv')
movies.head(7)


# In[20]:


####  Subtask 1.2: Inspect the Dataframe

##Inspect the dataframe for dimensions, null-values, and summary of different numeric columns.


# In[22]:


#Check the number of rows and columns in dataframe.

movies.shape


# In[23]:


#Check the column-wise info of the dataframe.

movies.info()


# In[24]:


#Check the summary for the numeric values

movies.describe()


# In[25]:


### Task 2: Data Analysis

### Now let's start with some data manipulation,  data analysis, and visualisation to get various insights. 


# In[26]:


####  Subtask 2.1: Reduce those Digits!

###These numbers in the `budget` and `gross` are too big, compromising its readability. 
###Let's convert the unit of the `budget` and `gross` columns from `$` to `million $` first.


# In[27]:


## Divide the 'gross' and 'budget' column by 1000000 to convert '$' to 'million $'

movies['Gross'] = movies['Gross'] / 1000000
movies['budget'] = movies['budget'] / 1000000


# In[28]:


movies.head()


# In[29]:


####  Subtask 2.2: Let's Talk Profit!

#        1. Create a new column called `profit` which contains the difference of the two columns: `gross` and `budget`.
##       2. Sort the dataframe using the `profit` column as reference.
###      3. Extract the top ten profiting movies in descending order and store them in a new dataframe - `top10`.
####     4. Plot a scatter or a joint plot between the columns `budget` and `profit` and write a few words on what you observed.
#####    5. Extract the movies with a negative profit and store them in a new dataframe - `loss_making_movies`


# In[32]:


# Creat a new column with name profit by subtracting budget from gross

movies['Profit'] = movies['Gross'] - movies['budget']
movies[['Profit', 'Gross', 'budget']] 


# In[33]:


#Sort the dataframe 'Profit' column as a reference using the sort_values function. Make sure to set the argument  
#'ascending' to 'False'

movies = movies.sort_values(by='Profit', ascending=False)
movies[['Title', 'Profit']].head()


# In[34]:


# Get the top 10 profitable movies by using position based indexing. Specify the rows till 10 (0-9)

movies.head(10)


# In[37]:


#Plot Budget vs Profit
plt.scatter(movies['budget'],movies['Profit'], marker ='o', c='red')
plt.title('Budget vs Profit', fontdict={'fontsize':15, 'color':'black'} )
plt.xlabel("Budget")
plt.ylabel('Profit')
plt.show()


# In[38]:


##The dataset contains the 100 best performing movies from the year 2010 to 2016. 
##However scatter plot tells a different story. You can notice that there are some movies with negative profit. 
##Although good movies do incur losses, but there appear to be quite a few movie with losses. 
##What can be the reason behind this? Lets have a closer look at this by finding the movies with negative profit.


# In[41]:


#Find the movies with negative profits

loss_making_movies = movies[movies['Profit']<0]
loss_making_movies.head()


# In[42]:


### Subtask 2.3: The General Audience and the Critics

## You might have noticed the column `MetaCritic` in this dataset. This is a very popular website where an average score is determined through the scores given by the top-rated critics. Second, you also have another column `IMDb_rating` which tells you the IMDb rating of a movie. This rating is determined by taking the average of hundred-thousands of ratings from the general audience. 

## As a part of this subtask, we are required to find out the highest rated movies which have been liked by critics and audiences alike.
#1. `MetaCritic` score is on a scale of `100` whereas the `IMDb_rating` is on a scale of 10. Lets convert the `MetaCritic` column to a scale of 10.
#2. Now, to find out the movies which have been liked by both critics and audiences alike and also have a high rating overall, we need to -
#    - Create a new column `Avg_rating` which will have the average of the `MetaCritic` and `Rating` columns
#    - Retain only the movies in which the absolute difference(using abs() function) between the `IMDb_rating` and `Metacritic` columns is less than 0.5. 
#    - Sort these values in a descending order of `Avg_rating` and retain only the movies with a rating equal to higher than `8` and store these movies in a new dataframe `UniversalAcclaim`.


# In[45]:


# Change the scale of Metacritic

movies['MetaCritic'] = movies['MetaCritic']/10


# In[44]:


movies.head()


# In[104]:


# Find average ratings

movies['Avg_rating'] = (movies['MetaCritic'] + movies['IMDb_rating'])/2
movies['Avg_rating']


# In[50]:


df = movies[abs(movies['MetaCritic'] - movies['IMDb_rating']) < 0.5]


# In[51]:


# Sort in descending order of average rating

df1 = df[['Title','MetaCritic','IMDb_rating','Avg_rating']]


# In[52]:


df1 = df1.sort_values(by='Avg_rating', ascending=False)


# In[56]:


# Find the movie with MetaCritic and IMDb_rating < 0.5 and also with the average rating > 8

UniversalAcclaim = df1[df1['Avg_rating']>=8]
UniversalAcclaim


# In[57]:


#### Subtask 2.4: Find the Most Popular Trios - I

### A producer is looking to make a blockbuster movie. 
# There will primarily be three lead roles in his movie and he wish to cast the most popular actors for it. 
# Now, since he don't want to take a risk, he will cast a trio which has already acted in together in a movie before. 
# He wants us to find the most popular trio based on the Facebook likes of each of these actors.

## The dataframe has three columns to help us out for the same, 
# viz. `actor_1_facebook_likes`, `actor_2_facebook_likes`, and `actor_3_facebook_likes`. 
# Our objective is to find the trios which has the most number of Facebook likes combined. 
# That is, the sum of `actor_1_facebook_likes`, `actor_2_facebook_likes` and `actor_3_facebook_likes` should be maximum.

## Lets Find out the top 5 popular trios, and output their names in a list.


# In[58]:


# Lets make a pivot table for The Most Popular Trios

group = movies.pivot_table(values=["actor_1_facebook_likes","actor_2_facebook_likes","actor_3_facebook_likes"], aggfunc='sum', index=["actor_1_name","actor_2_name","actor_3_name"])


# In[59]:


group


# In[60]:


# Lets make a new column 'Total_likes' in group

group['Total_likes'] = group['actor_1_facebook_likes']+group['actor_2_facebook_likes']+group['actor_3_facebook_likes']
group


# In[61]:


# Lets sort them in descending order by Total_likes

group.sort_values(by='Total_likes',ascending=False,inplace=True)


# In[62]:


group


# In[64]:


# Lets get the Top 5 Trios

group.head(5)
#or
## or group.iloc[0:5,:]


# In[65]:


#### Subtask 2.5: Find the Most Popular Trios - II

#In the previous subtask we found the popular trio based on the total number of facebook likes. Let's add a small condition to it and make sure that all three actors are popular. The condition is **none of the three actors' Facebook likes should be less than half of the other two**. For example, the following is a valid combo:
#- actor_1_facebook_likes: 70000
#- actor_2_facebook_likes: 40000
#- actor_3_facebook_likes: 50000

#But the below one is not:
#- actor_1_facebook_likes: 70000
#- actor_2_facebook_likes: 40000
#- actor_3_facebook_likes: 30000

#since in this case, `actor_3_facebook_likes` is 30000, which is less than half of `actor_1_facebook_likes`.

#Having this condition ensures that we aren't getting any unpopular actor in our trio (since the total likes calculated in the previous question doesn't tell anything about the individual popularities of each actor in the trio.).

#we can do a manual inspection of the top 5 popular trios we have found in the previous subtask and check how many of those trios satisfy this condition. Also, which is the most popular trio after applying the condition above?


# In[67]:


group = group[(group['actor_1_facebook_likes']>=group['actor_2_facebook_likes']/2) &
              (group['actor_1_facebook_likes']>=group['actor_3_facebook_likes']/2) &
              (group['actor_2_facebook_likes']>=group['actor_1_facebook_likes']/2)&
              (group['actor_2_facebook_likes']>=group['actor_3_facebook_likes']/2)&
              (group['actor_3_facebook_likes']>=group['actor_1_facebook_likes']/2)&
              (group['actor_3_facebook_likes']>=group['actor_2_facebook_likes']/2)]


# In[68]:


group


# In[69]:


group.head(5)


# In[70]:


#### Subtask 2.6: Runtime Analysis

# There is a column named `Runtime` in the dataframe which primarily shows the length of the movie. 
# It might be intersting to see how this variable this distributed. 
# Plot a `histogram` or `distplot` of seaborn to find the `Runtime` range most of the movies fall into.


# In[71]:


df = pd.DataFrame(group)


# In[86]:


column_name = movies.columns['Runtime']
column = df[column_name]
print(column)


# In[74]:


plt.hist(movies['Runtime'])
plt.show()


# In[75]:


sns.distplot(movies['Runtime'])


# In[76]:


#### Subtask 2.7: R-Rated Movies

# Although R rated movies are restricted movies for the under 18 age group, still there are vote counts from that age group. 
# Among all the R rated movies that have been voted by the under-18 age group, 

#lets find the top 10 movies that have the highest number of votes i.e.`CVotesU18` from the `movies` dataframe. 
#Store these in a dataframe named `PopularR`.


# In[78]:


popularR = movies[movies["content_rating"]== 'R'].sort_values(by='CVotesU18',ascending=False)


# In[79]:


popularR[['Title','content_rating','CVotesU18']].head(5)


# In[80]:


### Task 3 : Demographic analysis

# If we take a look at the last columns in the dataframe, most of these are related to demographics of the voters. 
# We also have three genre columns indicating the genres of a particular movie. 
# We will extensively use these columns for the third and the final stage of our assignment wherein we will analyse the voters across all demographics and also see how these vary across various genres. 
# let's get started with `demographic analysis`.


# In[81]:


####  Subtask 3.1 Combine the Dataframe by Genres

# There are 3 columns in the dataframe - `genre_1`, `genre_2`, and `genre_3`. As a part of this subtask, we need to aggregate a few values over these 3 columns. 
#1. First create a new dataframe `df_by_genre` that contains `genre_1`, `genre_2`, and `genre_3` and all the columns related to **CVotes/Votes** from the `movies` data frame. There are 47 columns to be extracted in total.
#2. Now, Add a column called `cnt` to the dataframe `df_by_genre` and initialize it to one. You will realise the use of this column by the end of this subtask.
#3. First group the dataframe `df_by_genre` by `genre_1` and find the sum of all the numeric columns such as `cnt`, columns related to CVotes and Votes columns and store it in a dataframe `df_by_g1`.
#4. Perform the same operation for `genre_2` and `genre_3` and store it dataframes `df_by_g2` and `df_by_g3` respectively. 
#5. Now that you have 3 dataframes performed by grouping over `genre_1`, `genre_2`, and `genre_3` separately, it's time to combine them. For this, add the three dataframes and store it in a new dataframe `df_add`, so that the corresponding values of Votes/CVotes get added for each genre.There is a function called `add()` in pandas which lets you do this. You can refer to this link to see how this function works. https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.add.html
#6. The column `cnt` on aggregation has basically kept the track of the number of occurences of each genre.Subset the genres that have atleast 10 movies into a new dataframe `genre_top10` based on the `cnt` column value.
#7. Now, take the mean of all the numeric columns by dividing them with the column value `cnt` and store it back to the same dataframe. We will be using this dataframe for further analysis in this task unless it is explicitly mentioned to use the dataframe `movies`.
#8. Since the number of votes can't be a fraction, type cast all the CVotes related columns to integers. Also, round off all the Votes related columns upto two digits after the decimal point.


# In[82]:


# Create the dataframe df_by_genre
demographics = movies.columns[11:-2]
demographics


# In[87]:


df_by_genre = movies[demographics]
df_by_genre.head()


# In[88]:


# Creat a column and initialize it to 1

df_by_genre["cnt"]=1


# In[89]:


# Group the movies by individual genres

df_by_g1 = df_by_genre.groupby('genre_1').sum()
df_by_g1


# In[90]:


df_by_g2 = df_by_genre.groupby('genre_2').sum()
df_by_g2


# In[92]:


df_by_g3 = df_by_genre.groupby('genre_3').sum()
df_by_g3


# In[99]:


# Add the grouped DataFrame and stored it in a new DataFrame

df.add = df_by_g1.add(df_by_g2, fill_value=0).add(df_by_g3, fill_value=0)
df.add


# In[103]:


# Extract genres with atleast 10 occurences
genre_top10 = df_add[df_add["cnt"]>=1].head(10)
genre_top10
cnt = genre_top10['cnt']
genre_top10 = genre_top10.drop("cnt", axis=1)


# In[105]:


df.dropna(subset=['Title', 'title_year', 'IMDb_rating', 'genre_1'], inplace=True)


# In[106]:


print("Data Overview:")
print(df.head())


# In[107]:


print("\nDescriptive Statistics:")
print(df.describe())


# In[113]:


plt.figure(figsize=(10, 6))
plt.scatter(movies['budget'], movies['Gross'], color='b', alpha=0.5)
plt.title('Budget vs Gross')
plt.xlabel('Budget (in millions)')
plt.ylabel('Gross (in millions)')
plt.show()


# In[111]:


plt.figure(figsize=(10, 6))
# plt.hist(movies['Runtime'])
plt.hist(movies['IMDb_rating'], bins=10, color='g', alpha=0.7)
plt.title('IMDb Rating Distribution')
plt.xlabel('IMDb Rating')
plt.ylabel('Frequency')
plt.show()


# In[114]:


# Data visualization - Genre distribution
plt.figure(figsize=(12, 6))
genre_counts = movies['genre_1'].value_counts()
genre_counts.plot(kind='bar', color='m', alpha=0.7)
plt.title('Genre Distribution')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()


# In[115]:


# Data visualization - Country distribution
plt.figure(figsize=(12, 6))
country_counts = movies['Country'].value_counts().head(10)
country_counts.plot(kind='bar', color='c', alpha=0.7)
plt.title('Top 10 Countries with Most Movies')
plt.xlabel('Country')
plt.ylabel('Count')
plt.show()


# In[116]:


def hypothesis_testing(data1,data2):
  alpha = [0.01, 0.05, 0.1] #common values for significance level
  from scipy.stats import ttest_ind
  stat,p_value = ttest_ind(data1,data2, equal_var= False)
  print(p_value/2)#since we are doing single tail test, we need only p_value/2
  for i in alpha:
    if p_value/2 < i:
      print('Reject the Null Hypothesis H0 at {} % significance level'.format(i*100))
      break
    else:
      print('Failed to reject the Null Hypothesis H0 at {} % significance level'.format(i*100))


# In[118]:


hypothesis_testing(movies['profit'],movies['profit'])


# In[119]:


def hypothesis_testing(data1,data2):
  alpha = [0.01, 0.05, 0.1] #common values for significance level
  from scipy.stats import ttest_ind
  stat,p_value = ttest_ind(data1,data2, equal_var= False)
  print(p_value/2)#since we are doing single tail test, we need only p_value/2
  for i in alpha:
    if p_value/2 < i:
      print('Reject the Null Hypothesis H0 at {} % significance level'.format(i*100))
      break
    else:
      print('Failed to reject the Null Hypothesis H0 at {} % significance level'.format(i*100))


# In[120]:


hypothesis_testing(movies['Profit'])


# In[ ]:




