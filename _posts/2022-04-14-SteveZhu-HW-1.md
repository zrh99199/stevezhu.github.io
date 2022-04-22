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
                             " for stations in " + country + "for years " + str(year_begin) + " - " + str(year_end),
                             **kwargs)
```

Show the plot
```python
color_map = px.colors.diverging.RdGy_r # choose a colormap

fig = temperature_coefficient_plot("India", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   width = 1000,
                                   height = 500,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map)

fig.show()
```
{% include temp_coef_plot.html %}

Now, I want to know which countries have the largest standard deviation in temperature?(Or which part's temperature varys more often than others)
To find out the answer, I want to build a choropleths

*Write a function to get a data from the database*
```python
def query_climate_database_2(year_begin, month):
    """
    A function take input year_begin" and "month", return a dataframe contains 
    stations' name, latitude, longitude, country, year, month and temp
    """
    
    conn = sqlite3.connect("hw1.db")
    cmd = \
    """
    SELECT S.name, S.latitude, S.longitude, C.country, T.year, T.month, T.temp \
    FROM temperatures T \
    LEFT JOIN countries C ON SUBSTR(T.id,1,2) == C.FIPS_10_4 \
    LEFT JOIN stations S ON T.id = S.id \
    WHERE T.year >= ? 
    AND month = ?
    """
    
    return pd.read_sql(cmd, conn, params=[year_begin, month])
```
Check if it works

```python
query_climate_database_2(1980, 1)
```

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
      <td>SAVE</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>Antigua and Barbuda</td>
      <td>1980</td>
      <td>1</td>
      <td>-2.71</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SAVE</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>Antigua and Barbuda</td>
      <td>1981</td>
      <td>1</td>
      <td>-1.01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SAVE</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>Antigua and Barbuda</td>
      <td>1982</td>
      <td>1</td>
      <td>-5.51</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SAVE</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>Antigua and Barbuda</td>
      <td>1983</td>
      <td>1</td>
      <td>4.69</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SAVE</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>Antigua and Barbuda</td>
      <td>1984</td>
      <td>1</td>
      <td>0.09</td>
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
      <th>561455</th>
      <td>BEITBRIDGE</td>
      <td>-22.2170</td>
      <td>30.0000</td>
      <td>Zimbabwe</td>
      <td>1989</td>
      <td>1</td>
      <td>28.45</td>
    </tr>
    <tr>
      <th>561456</th>
      <td>BEITBRIDGE</td>
      <td>-22.2170</td>
      <td>30.0000</td>
      <td>Zimbabwe</td>
      <td>1990</td>
      <td>1</td>
      <td>27.56</td>
    </tr>
    <tr>
      <th>561457</th>
      <td>HARARE_BELVEDERE</td>
      <td>-17.8300</td>
      <td>31.0200</td>
      <td>Zimbabwe</td>
      <td>1983</td>
      <td>1</td>
      <td>22.60</td>
    </tr>
    <tr>
      <th>561458</th>
      <td>HARARE_BELVEDERE</td>
      <td>-17.8300</td>
      <td>31.0200</td>
      <td>Zimbabwe</td>
      <td>1984</td>
      <td>1</td>
      <td>21.70</td>
    </tr>
    <tr>
      <th>561459</th>
      <td>GRAND_REEF</td>
      <td>-18.9800</td>
      <td>32.4500</td>
      <td>Zimbabwe</td>
      <td>1981</td>
      <td>1</td>
      <td>24.30</td>
    </tr>
  </tbody>
</table>
<p>561460 rows × 7 columns</p>
</div>

*Write the plot function*
```python
import numpy as np

def temp_std_plot(year_begin, month, min_obs, **kwargs):
    """
    A function take input "country", "year_begin", "month", "min_obs" and "**kwargs"
    return a scatter geo to show the standard deviation of temperature for the given month
    """
    
    df = query_climate_database_2(year_begin, month)

    # drow the row with less than min_obs observations
    name_col = df[['NAME', 'Temp']].copy()
    name_col['Num_obs'] = name_col.groupby(["NAME"]).transform(len)
    df['Num_obs'] = name_col['Num_obs']
    df = df[df['Num_obs'] >= min_obs]
    
    # create graph data frame
    graph_df = df[['NAME','LATITUDE','LONGITUDE','country']].drop_duplicates(subset = ["NAME"])

    # calculate the std of temp by groupby
    df_std = df.groupby(["NAME","Month"])["Temp"].aggregate(np.std).to_frame(name='std')
    df_std['std'] = round(df_std['std'], 4)
    
    # merge two data frames
    graph_df = graph_df.merge(df_std, left_on='NAME', right_on='NAME')

    return px.scatter_geo(graph_df, lat="LATITUDE", lon = "LONGITUDE",
                          hover_name="NAME", 
                          size="std",
                          title = "Standard deviation of temperature in " + pd.to_datetime(month, format='%m').month_name() + 
                             " since " + str(year_begin),
                          **kwargs)
```

*Show the plot*

```python
fig = temp_std_plot(1980, 1, min_obs = 10, size_max=5, projection="natural earth")
fig.show()
```
{% include temp_std_plot.html %}

From the plot, I found in January, the places in high latitude tend to have higher standard deviation of temperature than the places in low latitude

So, plot the latitude with std to see their relationship

*Write the plot function*
```python
import seaborn as sns
import matplotlib.pyplot as plt

def lat_std_scplot(year_begin, month, min_obs, **kwargs):
    """
    A function take input "country", "year_begin", "month", "min_obs" and "**kwargs"
    return a scatter plot to show the relationship between standard deviation of temperature 
    and latitude for the given month
    """
    
    # prepare data frame as same as previous plot
    df = query_climate_database_2(year_begin, month)

    # drow the row with less than min_obs observations
    name_col = df[['NAME', 'Temp']].copy()
    name_col['Num_obs'] = name_col.groupby(["NAME"]).transform(len)
    df['Num_obs'] = name_col['Num_obs']
    df = df[df['Num_obs'] >= min_obs]
    
    # create graph data frame
    graph_df = df[['NAME','LATITUDE','LONGITUDE','country']].drop_duplicates(subset = ["NAME"])

    # calculate the std of temp by groupby
    df_std = df.groupby(["NAME","Month"])["Temp"].aggregate(np.std).to_frame(name='std')
    df_std['std'] = round(df_std['std'], 4)
    
    # merge two data frames
    graph_df = graph_df.merge(df_std, left_on='NAME', right_on='NAME')
    
    ax = sns.lmplot(x='LATITUDE', y='std', data=graph_df, **kwargs)
    ax.set(ylabel = "Standard deviation of temperature in "+ pd.to_datetime(month, format='%m').month_name(), xlabel = "Latitude", title = "Relationship between Standard deviation of temperature and latitude")
    ax.fig.set_figwidth(10)
    ax.fig.set_figheight(5)
```

*Show the plot*
```python
lat_std_scplot(1980, 1, min_obs = 10, ci = None, scatter_kws = {"s": 10}, line_kws ={"color": "red"})
```
![Hw1_Sctplot.png](/images/Hw1_Sctplot.png)

Based on the scatter plot, I found there is a positive correlation between the latitude and the standard deviation of temeprature, which means a place in high latitude tends to exprience more variable temperature in January

### Question 2: Visualize how temperature changes for each month for a country for a given period of time

*Write a function to get data from the database*

```python
def query_climate_database_3(country, year_begin, year_end):
    """
    A function take input country, year_begin and year_end, return a 
    dataframe contains country, year, month and average of temperature
    """
    
    conn = sqlite3.connect("hw1.db")
    cmd = \
    """
    SELECT T.year, T.month, AVG(T.temp) mean_temp \
    FROM temperatures T \
    LEFT JOIN countries C ON SUBSTR(T.id,1,2) == C.FIPS_10_4 \
    WHERE country = ?
    AND year >= ?
    AND year <= ?
    GROUP BY year, month
    """
    
    return pd.read_sql(cmd, conn, params=[country, year_begin, year_end]) 
```
   
*Write the plot function*

```python
import calendar

def temp_heatmap(country, year_begin, year_end, **kwargs):
    """
    A function take input country, year_begin, year_end and kwargs,
    return a headmap plot
    """
    df = query_climate_database_3(country, year_begin, year_end)
    #tranform the data frame into pivot layout
    graph_df = df.pivot("Year", "Month", "mean_temp")

    #reverse the order of the data frame
    graph_df = graph_df.iloc[::-1]

    #rename the columns by using English words for months
    dd=dict((enumerate(calendar.month_abbr)))
    graph_df = graph_df.rename(columns=dd,level=0)

    fig = px.imshow(graph_df, 
                    labels = dict(color = "Temperature(°C)"), 
                    x =graph_df.columns, 
                    y =graph_df.index,
                    title = "Heatmap of Temperature(°C) in " + country + " bewteen " + str(year_begin) + " and " + str(year_end) + " for each month",
                    **kwargs)
    fig.update_layout(width = 1000, height = 1000)
    return fig
```
*Show the plot*

```python
fig = temp_heatmap("Brazil", 1980, 2020, color_continuous_scale = 'icefire')
fig.show()
```

{% include temp_heatmap.html %}

### Question 3: Which parts(North-East, North-West, South-East, South-West) of a given country have a higher average yearly change of temperature in a given month?

*Write the plot function*

I first categorize stations into four categories based on their latitude and longtitude, then use boxplot to visualize them. (Histogram and violin plot will be good too)

```python
def temp_change_boxplot(country, year_begin, year_end, month, **kwargs):
    """
    A function take input "country", "year_begin", "year_end", "month" and "**kwargs"
    return histograms to show the Estimated Yearly Increase in Temperature for the
    given country, year and month
    """
    df = query_climate_database(country, year_begin, year_end, month)

    # add columns contains stations' geological information to the data frame 
    long_range = max(df["LONGITUDE"])-min(df["LONGITUDE"])
    min_long = min(df["LONGITUDE"])
    df['EW'] = ""
    df.loc[df["LONGITUDE"] < (min_long + long_range/2), ['EW']] = 'West'
    df.loc[df["LONGITUDE"] >= (min_long + long_range/2), ['EW']] = 'East'

    lat_range = max(df["LATITUDE"])-min(df["LATITUDE"])
    min_lat = min(df["LATITUDE"])
    df['NS'] = ""
    df.loc[df["LATITUDE"] < (min_lat + lat_range/2), ['NS']] = 'South'
    df.loc[df["LATITUDE"] >= (min_lat + lat_range/2), ['NS']] = 'North'
    
    # create a new data frame contains only name, geological information and Estimated Yearly Increase in Temperature (°C)
    graph_df = df.groupby(["EW","NS","NAME"]).apply(average_increase).reset_index(name="Estimated Yearly Increase in Temperature (°C)")

    return px.box(data_frame = graph_df,
                        x = "Estimated Yearly Increase in Temperature (°C)",
                        facet_row = "EW",
                        facet_col = "NS",
                        title = "Boxplot of estimates of yearly increase in temperature in " + pd.to_datetime(month, format='%m').month_name() + " for stations in " + country + " for years " + str(year_begin) + " - " + str(year_end),
                        **kwargs)

```
*Show the plot*

```python
fig = temp_change_boxplot("India", 1980, 2020, 1, width = 500, height = 1000)
fig.show()
```
{% include temp_box_plot.html %}

Based on the box plot, I found in India, the South-West part has more serious climate change problem than other parts.