---
layout: post
title: Homework 0
---
# Task: Write a tutorial explaining how to construct an interesting data visualization of the Palmer Penguins data set.

## 1. Load the data

Import pandas and use pandas to load the csv file from the internet
```
import pandas as pd
url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/palmer_penguins.csv"
penguins = pd.read_csv(url)
```