---
layout: post
title: Homework 0
---
## Task: Write a tutorial explaining how to construct an interesting data visualization of the Palmer Penguins data set.

### 1. Load and clean the data

Import pandas and use pandas to load the csv file from the internet
```
import pandas as pd
url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/palmer_penguins.csv"
penguins = pd.read_csv(url)
```
Clean the data
```
penguins = penguins.dropna(subset = ["Body Mass (g)", "Sex"])
penguins["Species"] = penguins["Species"].str.split().str.get(0)
penguins = penguins[penguins["Sex"] != "."]

cols = ["Species", "Island", "Sex", "Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Body Mass (g)"]
penguins = penguins[cols]
```
Take a look now
```
penguins.head()
```

### 2. Visualization

Import express from plotly and use express modules to visualize the features

```
from matplotlib import pyplot as plt
```

#### Scatter Plots
reference: https://plotly.com/python-api-reference/generated/plotly.express.scatter.html
```
fig = px.scatter(data_frame = penguins, # data that needs to be plotted
                 x = "Culmen Length (mm)", # column name for x-axis
                 y = "Culmen Depth (mm)", # column name for y-axis
                 color = "Species", # column name for 
                 width = 500,
                 height = 300)

# reduce whitespace
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# show the plot
fig.show()
```