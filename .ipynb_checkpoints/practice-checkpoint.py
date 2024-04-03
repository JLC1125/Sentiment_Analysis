# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:45:29 2024

@author: Levin
"""

#import libraries
import requests
import pandas as pd
import time
import psycopg2 as ps
from dotenv import load_dotenv
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.ticker import FuncFormatter


load_dotenv(r"C:\Users\Levin\Projects\.env")

#connect to db
def connect_to_db(host_name, dbname, port, username, password):
    try:
        conn = ps.connect(host=host_name, database=dbname, user=username, password=password, port=port)
    except ps.OperationalError as e:
            raise e
    else:
        print('Connected!')
    return conn

host_name = os.getenv('host_name')
dbname = os.getenv('dbname')
port = os.getenv('port')
username = os.getenv('un')
password = os.getenv('password')
conn = None

conn = connect_to_db(host_name, dbname, port, username, password)


query = """
SELECT * FROM videos;
"""
df = pd.read_sql_query(query, conn)
df1 = df.copy()


df1.set_index('upload_date', inplace=True)
df1['view_count'].plot()

plt.title('View Count Over Time')
plt.xlabel('upload_date')
plt.ylabel('view_count')
plt.show()


df['like_count'].plot()

plt.title('Like Count Over Time')
plt.xlabel('upload_date')
plt.ylabel('like_count')
plt.show()



from flair.models import TextClassifier
from flair.data import Sentence

classifier = TextClassifier.load("en-sentiment")
pos_sentence = Sentence("I really like Flair!")
neg_sentence = Sentence("Flair is bad!")
classifier.predict(pos_sentence)
classifier.predict(neg_sentence)
print(pos_sentence.labels, neg_sentence.labels)


def add_sentiment(df):
    title_sentiment = []

    for i,vid in df.iterrows():
        title = Sentence(vid['video_title'])
        classifier.predict(title)
        for label in title.labels:
            if label.score <= .9:
                title_sentiment.append("NEUTRAL")
            else:
                title_sentiment.append(label.value)

    df["title_sentiment"] = title_sentiment
    
add_sentiment(df)

df2=df.copy()


sorted_data = df2.sort_values(by='view_count', ascending=False)

# Rename the columns
renamed_data = sorted_data.rename(columns={'video_title': 'Title', 'view_count': 'Views', 'title_sentiment': 'Sentiment (Modeled)'})

# Select specific columns
selected_data = renamed_data[['Title', 'Views', 'Sentiment (Modeled)']]

# Filter rows where 'Views' are greater than or equal to the Views in the 10th row
filtered_data = selected_data[selected_data['Views'] >= selected_data['Views'].iloc[9]]


print(filtered_data.style)

sorted_data2 = df2.sort_values(by='comment_count', ascending=False)

# Rename the columns
renamed_data2 = sorted_data2.rename(columns={'video_title': 'Title', 'comment_count': 'Comments', 'title_sentiment': 'Sentiment (Modeled)'})

# Select specific columns
selected_data2 = renamed_data2[['Title', 'Comments', 'Sentiment (Modeled)']]

# Filter rows where 'Views' are greater than or equal to the Views in the 10th row
filtered_data2 = selected_data2[selected_data2['Comments'] >= selected_data2['Comments'].iloc[9]]


print(filtered_data2.style)

sorted_data3 = df2.sort_values(by='like_count', ascending=False)

# Rename the columns
renamed_data3 = sorted_data3.rename(columns={'video_title': 'Title', 'like_count': 'Likes', 'title_sentiment': 'Sentiment (Modeled)'})

# Select specific columns
selected_data3 = renamed_data3[['Title', 'Likes', 'Sentiment (Modeled)']]

# Filter rows where 'Views' are greater than or equal to the Views in the 10th row
filtered_data3 = selected_data3[selected_data3['Likes'] >= selected_data3['Likes'].iloc[9]]


print(filtered_data3.style)



# Assuming 'video_data' is your DataFrame
# Select and rename columns
selected_data = df2[['video_title', 'view_count', 'like_count', 'comment_count', 'upload_date']].rename(columns={'view_count': 'Views', 'like_count': 'Likes', 'comment_count': 'Comments'})

# Pivot the DataFrame to a long format
long_data = pd.melt(selected_data, id_vars=['video_title', 'upload_date'], value_vars=['Views', 'Likes', 'Comments'], var_name='Metric', value_name='Value')

# Plotting
sns.set_theme(style="whitegrid")  # Setting the seaborn theme. You can adjust this for a 'fresh' appearance.

g = sns.relplot(
    data=long_data,
    x='upload_date', y='Value', col='Metric', kind='line',
    col_wrap=1, height=5, aspect=1.5, facet_kws=dict(sharex=False, sharey=False)
)

# Adding points to the line plot
g.map(sns.scatterplot, 'upload_date', 'Value', s=10)

# Customizing the facets
g.set_titles("{col_name}", size=12, weight='bold')
g.set_axis_labels("Upload Date", "")
g.set_xticklabels(rotation=45, ha="right")

# Adjust y-axis formatting to include commas in labels for thousands
for ax in g.axes.flatten():
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.show()



    

