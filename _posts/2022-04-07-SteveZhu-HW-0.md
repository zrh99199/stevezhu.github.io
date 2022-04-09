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

```
  Species     Island     Sex  Culmen Length (mm)  Culmen Depth (mm)  \
0  Adelie  Torgersen    MALE                39.1               18.7   
1  Adelie  Torgersen  FEMALE                39.5               17.4   
2  Adelie  Torgersen  FEMALE                40.3               18.0   
4  Adelie  Torgersen  FEMALE                36.7               19.3   
5  Adelie  Torgersen    MALE                39.3               20.6   

   Flipper Length (mm)  Body Mass (g)  
0                181.0         3750.0  
1                186.0         3800.0  
2                195.0         3250.0  
4                193.0         3450.0  
5                190.0         3650.0    
```

### 2. Visualization

We will learn two ways to visualize our data, using matplotlib and using plotly.

#### matplotlib
Import pyplot from matplotlib and use pyplot modules to visualize the features
```python
from matplotlib import pyplot as plt
```

#### *Scatter plots*

```python
fig,ax = plt.subplots(1) #set the plot first

for g in penguins['Species'].unique():
    ax.scatter(penguins['Culmen Length (mm)'][penguins['Species'] == g], 
               penguins['Culmen Depth (mm)'][penguins['Species'] == g],
               label = g) #label the point by their species, easy to add legend later
    
ax.set_xlabel("Culmen Length (mm)", fontsize = 20) #set x-axis label
ax.set_ylabel("Culmen Depth (mm)", fontsize = 20) #set y-axis label
fig.suptitle("Relationship between Culmen Length and Culmen Depth") #set title    
ax.legend() #add legend
```
![HW0-plot1.png](/images/HW0-plot3.png)

#### *Histograms*

```python
fig,ax = plt.subplots(1) #set the plot first

for g in penguins['Species'].unique():
    ax.hist(penguins['Body Mass (g)'][penguins['Species'] == g],
            alpha = 0.5, #alpha control the opacity of the graph
            bins = 10, #bin numbers
            label = g) #label the point by their species, easy to add legend later
    
ax.set_xlabel("Body Mass(g)", fontsize = 20) #set x-axis label
ax.set_ylabel("Frequency", fontsize = 15) #set y-axis label

fig.suptitle("Histogram of body mass by species") #set title    
ax.legend() #add legend
```
![HW0-plot1.png](/images/HW0-plot4.png)

#### plotly
plotly is a easier way to create plots, and the created plot is interactable.

Import express from plotly and use express modules to visualize the features
```python
from plotly import express as px
```

#### *Scatter plots*
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

#### *Histograms*
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