---
layout: post
title: Homework 1
---
## Visualize climate change by using plotly

### Question 1: How does the average yearly change in temperature vary within a given country

In order to answer this question, I decide to make a Geographic Scatter Plot

*Import and create a database*
```python
import pandas as pd
import sqlite3

# read csv into python
temps = pd.read_csv("temps_stacked.csv")
countries = pd.read_csv("countries.csv")
stations = pd.read_csv("https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/noaa-ghcn/station-metadata.csv")

# rewrite some columns name
countries = countries.rename(columns= {"FIPS 10-4": "FIPS_10_4"})
countries = countries.rename(columns= {"Name": "country"})

# open a connection to hw1.db
conn = sqlite3.connect("hw1.db")

temps.to_sql("temperatures", conn, if_exists="replace", index=False)
countries.to_sql("countries", conn, if_exists="replace", index=False)
stations.to_sql("stations", conn, if_exists="replace", index=False)

# always close connection
conn.close()
```

*Write a function to get data from the database*

```python
def query_climate_database(country, year_begin, year_end, month):
    """
    A function take input "country", "year_begin", "year_end" and "month"
    return a dataframe contains stations' name, latitude, longitude, country, year, month and temp
    """
    
    conn = sqlite3.connect("hw1.db")
    cmd = \
    """
    SELECT S.name, S.latitude, S.longitude, C.country, T.year, T.month, T.temp \
    FROM temperatures T \
    LEFT JOIN countries C ON SUBSTR(T.id,1,2) == C.FIPS_10_4 \
    LEFT JOIN stations S ON T.id = S.id \
    WHERE C.country = ?
    AND T.year >= ?
    AND T.year <= ?
    AND T.month = ?
    """
    
    return pd.read_sql(cmd, conn, params=[country, year_begin, year_end, month]) # params represent the '?'s in cmd
```

Check if it works
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>country</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>6.90</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1994</td>
      <td>1</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>3150</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1995</td>
      <td>1</td>
      <td>5.60</td>
    </tr>
    <tr>
      <th>3151</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>5.70</td>
    </tr>
  </tbody>
</table>
<p>3152 rows × 7 columns</p>
</div>

It works! Now build a plot function based on the data frame

*Build the plot function*

First we need a function to calculate the linear regression coefficients, so we can use it to calculate the average yearly change in temperature
```python
from sklearn.linear_model import LinearRegression

def average_increase(df):
    """
    A function calculate a data frame's linear regression coefficients
    base on "Year" and "Temp" features
    """
    x = df[["Year"]] 
    y = df["Temp"]   
    LR = LinearRegression()
    LR.fit(x, y)
    return round(LR.coef_[0],4)
```

Now construct the plot function
```python
from plotly import express as px

def temperature_coefficient_plot(country, year_begin, year_end, month, min_obs, **kwargs):
    """
    A function take input "country", "year_begin", "year_end", "month", "min_obs" and "**kwargs"
    return a scatter mapbox to show the estimated yearly increase in temperature for the given 
    country, year and month
    """
    df = query_climate_database(country, year_begin, year_end, month)

    # drow the row with less than min_obs observations
    name_col = df[['NAME', 'Temp']].copy()
    name_col['Num_obs'] = name_col.groupby(["NAME"]).transform(len)
    df['Num_obs'] = name_col['Num_obs']
    df = df[df['Num_obs'] >= min_obs]
    
    # create a data frame which contains only the neccessary data for plotting
    graph_df = df[['NAME','LATITUDE','LONGITUDE','country']].drop_duplicates(subset = ["NAME"])
    
    # create a data frame contains the name and yearly increase in temperature
    df_coefs = df.groupby("NAME").apply(average_increase)
    df_coefs = df_coefs.to_frame(name='Estimated Yearly Increase in Temperature (°C)')
    
    # merge two data frames
    graph_df = graph_df.merge(df_coefs, left_on='NAME', right_on='NAME')
    
    # create our plot with scatter_mapbox
    return px.scatter_mapbox(graph_df, 
                             lat = "LATITUDE",
                             lon = "LONGITUDE",
                             hover_name = "NAME",
                             color = "Estimated Yearly Increase in Temperature (°C)",
                             title = "Estimates of yearly increase in temperature in " + pd.to_datetime(month, format='%m').month_name() + 
                             " for stations in " + country + ", years " + str(year_begin) + " - " + str(year_end),
                             **kwargs)
```

Let's try if it works
```python
color_map = px.colors.diverging.RdGy_r # choose a colormap

fig = temperature_coefficient_plot("India", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map)

fig.show()
```
file:///C:/Users/zrh99/Downloads/temp_coef_plot.html
