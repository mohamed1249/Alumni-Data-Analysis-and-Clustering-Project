# Alumni Data Analysis and Clustering Project

This repository contains code for data cleaning, exploratory data analysis (EDA), and machine learning clustering techniques applied to a dataset of alumni information to identify potential donors.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- openpyxl

## Project Overview

The project involves the following steps:

1. **Data Loading and Initial Exploration**
2. **Data Cleaning**
3. **Exploratory Data Analysis (EDA)**
4. **Clustering Analysis**

## Instructions

### 1. Data Loading and Initial Exploration

The dataset is loaded from an Excel file, and initial exploration is performed to understand the structure and summary statistics of the data.

```python
import numpy as np 
import pandas as pd 
pd.set_option('display.max_rows', 100)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('darkgrid')
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 

import warnings
warnings.filterwarnings("ignore")

data = pd.read_excel('BI Officer data set (test data) (3).xlsx', index_col='ID Number')
data.head()

data.info()
data.describe()
```

### 2. Data Cleaning

Cleaning steps include handling missing values, dropping irrelevant columns, and ensuring data consistency.

```python
# Checking the number of null values in each column
nulls = data.isna().sum()
nulls[nulls > 0].sort_values(ascending=False)

# Checking the number of unique values in each column
uniques = data.nunique()
uniques.sort_values()

# Identifying and handling invalid columns
column_unvalidation = nulls = data.isna().sum() / data.nunique()
unvalid_columns = column_unvalidation[column_unvalidation > 0].sort_values(ascending=False)
unvalid_columns

# Dropping irrelevant columns and cleaning necessary ones
data = data.drop(['Recognition Preference', 'Title', 'Solicit Pref', 'Phone', 'Email', 'Address', 'Postal Code', 'Company Name'], axis=1)
data['Year of Last Gift'].fillna(data['Year of Last Gift'].mode()[0], inplace=True)
data['Year of First Gift'].fillna(data['Year of First Gift'].mode()[0], inplace=True)
data['Bequest Gift Score'].fillna(data['Bequest Gift Score'].mean(), inplace=True)
data['Student Involvement'].fillna('Unknown', inplace=True)
data['Largest Gift'].fillna(data['Largest Gift'].median(), inplace=True)
data['Affinity Score'].fillna(data['Affinity Score'].mean(), inplace=True)

# Reviewing the cleaned data
data.head()
data.describe()
```

### 3. Exploratory Data Analysis (EDA)

Visualizations are created to explore the distribution of key variables and identify patterns.

```python
# Preferred Language distribution
data['Preferred Language'][data['Preferred Language'] == ' '] = 'Unknown'
def pie(col):
    return px.pie(data, col, title=col)
pie('Preferred Language')

# Gender distribution
data = data[data.Gender != 'U']
pie('Gender')

# Past Traveler distribution
pie('Past Traveler Y/N')

# Instances of Volunteering distribution
pie('# of Instances of Volunteering')

# Sunburst chart for percentages of volunteering instances by language and gender
def percentages(cols, size=(1000, 1000)):
    fig = px.sunburst(data, path=cols)
    fig.update_traces(textinfo='label+percent entry')
    fig.update_layout(autosize=False, width=size[0], height=size[1])
    fig.show()
percentages(['# of Instances of Volunteering', 'Preferred Language', 'Gender', 'Past Traveler Y/N'])

# Faculty of Graduation analysis
grouped = data.groupby(['# of Instances of Volunteering', 'Faculty of Graduation']).size().reset_index().rename(columns={0: 'count'})
px.bar(grouped, x='# of Instances of Volunteering', y='count', color='Faculty of Graduation', barmode='group', title='Number of Instances of Volunteering in each Faculty')

# Most frequent Faculty of Graduation
def most_frequent(col, top=25, data=data):
    vc = data[col].value_counts().head(top)
    return px.bar(x=vc.index, y=vc, color=vc)
most_frequent('Faculty of Graduation')

# Most frequent Province
most_frequent('Province')

# Most frequent City
most_frequent('City')

# Distribution of Years of Graduation
import warnings
warnings.filterwarnings('ignore')
def dist(col, length=4, data=data):
    l = len(data) * length / 10
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Histogram(x=data[col], name=col))
    mean = data[col].mean()
    med = data[col].median()
    mode = data[col].mode()[0]
    fig.add_trace(go.Line(x=[mean for _ in range(round(l))], y=list(range(round(l))), name=f'{col}\'s mean'))
    fig.add_trace(go.Line(x=[med for _ in range(round(l))], y=list(range(round(l))), name=f'{col}\'s median'))
    fig.add_trace(go.Line(x=[mode for _ in range(round(l))], y=list(range(round(l))), name=f'{col}\'s mode'))
    fig.show()
dist('Year of Graduation', 0.25)
```

### 4. Clustering Analysis

Clustering techniques are applied to segment alumni based on their likelihood to donate.

```python
Q3_clustring_data = data[['Lifetime Giving', 'Year of Last Gift', 'Largest Gift', 'Events Attended in Lifetime', 'Affinity Score', 'Capacity Score', 'Bequest Gift Score', '# of Clicks in the Past Month']]
scaled_features = StandardScaler().fit_transform(Q3_clustring_data.values)
scaled_features_Q3_clustring_data = pd.DataFrame(scaled_features, index=Q3_clustring_data.index, columns=Q3_clustring_data.columns)

# Elbow method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_features_Q3_clustring_data)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(16, 8))
plt.plot(range(1, 11), wcss, 'bx-')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Silhouette scores to validate the number of clusters
silhouette_scores = []
for i in range(2, 10):
    m1 = KMeans(n_clusters=i, random_state=42)
    c = m1.fit_predict(scaled_features_Q3_clustring_data)
    silhouette_scores.append(silhouette_score(scaled_features_Q3_clustring_data, m1.fit_predict(scaled_features_Q3_clustring_data))) 
plt.bar(range(2, 10), silhouette_scores) 
plt.xlabel('Number of clusters', fontsize=20) 
plt.ylabel('S(i)', fontsize=20) 
plt.show()

# Training and predicting using K-Means Algorithm
kmeans = KMeans(n_clusters=2, random_state=42).fit(scaled_features_Q3_clustring_data)
pred = kmeans.predict(scaled_features_Q3_clustring_data)

# Appending cluster values to the main dataframe
data['cluster'] = pred + 1
data[data["cluster"] == 1].describe()
data[data["cluster"] == 2].describe()

# Visualizing the distribution of clusters
px.bar(x=data["cluster"].value_counts().index, y=data["cluster"].value_counts())
```

## Outputs

- `all sympathetics.xlsx`: Data of alumni identified as potential donors.
- `all not sympathetics.xlsx`: Data of alumni not identified as potential donors.
- `most likely to make a donation(sorted).xlsx`: Sorted data of alumni most likely to donate.
