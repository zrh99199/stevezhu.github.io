---
layout: post
title: Homework 0
---
## Task: Write a tutorial explaining how to construct an interesting data visualization of the Palmer Penguins data set.

### 1. Load and clean the data

Import pandas and use pandas to load the csv file from the internet
```python
import pandas as pd
url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/palmer_penguins.csv"
penguins = pd.read_csv(url)
```
Clean the data
```python
penguins = penguins.dropna(subset = ["Body Mass (g)", "Sex"])
penguins["Species"] = penguins["Species"].str.split().str.get(0)
penguins = penguins[penguins["Sex"] != "."]

cols = ["Species", "Island", "Sex", "Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Body Mass (g)"]
penguins = penguins[cols]
```
Take a look now
```python
penguins.head()
```

### 2. Visualization

We will learn two ways to visualize our data, using matplotlib and using plotly.

#### *matplotlib*

#### *plotly*
plotly is a easier way to create plots, and the created plot is interactable.

Import express from plotly and use express modules to visualize the features
```python
from plotly import express as px
```

#### Scatter Plots
reference: https://plotly.com/python-api-reference/generated/plotly.express.scatter.html
```python
fig = px.scatter(data_frame = penguins, # data that needs to be plotted
                 x = "Culmen Length (mm)", # column name for x-axis
                 y = "Culmen Depth (mm)", # column name for y-axis
                 color = "Species", # mark species by different colors
                 width = 500,
                 height = 300)

# reduce whitespace
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# show the plot
fig.show()
```
![HW0-plot1.png](/images/HW0-plot1.png)

#### Histograms
reference: https://plotly.com/python-api-reference/generated/plotly.express.histogram.html
```python
fig = px.histogram(data_frame = penguins, # data that needs to be plotted
                 x = "Culmen Length (mm)", # column name for x-axis
                 color = "Species", # column name for color coding
                 nbins = 50,
                 opacity = 0.7,
                 width = 600,
                 height = 400)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
```
![HW0-plot2.png](/images/HW0-plot2.png)

More plot types available on https://plotly.com/python-api-reference/generated/plotly.express.html#module-plotly.express