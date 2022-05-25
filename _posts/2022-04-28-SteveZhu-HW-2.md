---
layout: post
title: Homework 2 
---
## IMDB Web Scraping

In this blog, I will develop a web scraper to generate actor/actress data from IMDb, and use the data to give movies/TV shows recommendation based on my favorite movie *Interstellar*.

Here's a link to my project repository: https://github.com/zrh99199/HW2-IMDb-Web-Scraping/

### 1. Building web scraper

First, create a spider named imdb_spider with Interstellar imdb page(or any other movie imdb page) as start url 

```python
class ImdbSpider(scrapy.Spider):
    name = "imdb_spider"
    start_urls = ["https://www.imdb.com/title/tt0816692/"] #Movie Interstellar
```

Then, I develop the parse method which will redirect the movie/TV show page to its whole cast list
```python
    def parse(self, response):
    """
    redirect the start url to the fullcredits site
    """
        url = response.url + "fullcredits/"
        yield scrapy.Request(url, callback = self.parse_full_credit)
```

After getting in the cast page, the method parse_full_credit will redirect to all the actor/actress pages appeal in the cast page
```python
    def parse_full_credit(self, response):
    """
    redirect the fullcredits site to the actor/actress sites
    """
        prefix = "https://www.imdb.com"
        #get all the actor/actress imdb sites' suffix from td.primary_photo class
        suffixs = [a.attrib["href"] for a in response.css("td.primary_photo a")]
        urls = [prefix + suffix for suffix in suffixs]
        for url in urls:
            yield scrapy.Request(url, callback = self.parse_actor_page)
```

When we get in the actor/actress page, the method parse_actor_page will generate all the movies/TV shows he/she acts, and output the actor/actress' name with the movies/TV shows' name
```python
    def parse_actor_page(self,response):
    """
    generate all the movies and tv shows the actor/actress acts
    """
        #get the name of actor/actress from 'title' class
        actor = response.css('title').get()[7:].split(" - IMDb",1)[0]
        #only look for the div.filmo-row with id like 'actor-...' or 'actress-...'
        #get the name of the movie from the class div.filmo-row with id started with 'act'
        for movie_tv in response.css('div.filmo-row[id^="act"]'):
            yield {
                "actor": actor,
                "movie_tv": movie_tv.css("a::text").get()
            }
```

Run the following code and we will get a csv file contains a lot of actor/actress and movie/TV show information
```python
scrapy crawl imdb_spider -o result.csv
```

### 2. Analyze the spreadsheet

Read the result.csv into python and take a look

```python
import pandas as pd
df = pd.read_csv('result.csv')
df
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
      <th>actor</th>
      <th>movie_tv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>William Devane</td>
      <td>Bosch: Legacy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>William Devane</td>
      <td>The Grinder</td>
    </tr>
    <tr>
      <th>2</th>
      <td>William Devane</td>
      <td>Jesse Stone: Lost in Paradise</td>
    </tr>
    <tr>
      <th>3</th>
      <td>William Devane</td>
      <td>Truth</td>
    </tr>
    <tr>
      <th>4</th>
      <td>William Devane</td>
      <td>Interstellar</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2389</th>
      <td>Jessica Chastain</td>
      <td>Close to Home</td>
    </tr>
    <tr>
      <th>2390</th>
      <td>Jessica Chastain</td>
      <td>Law &amp; Order: Trial by Jury</td>
    </tr>
    <tr>
      <th>2391</th>
      <td>Jessica Chastain</td>
      <td>Dark Shadows</td>
    </tr>
    <tr>
      <th>2392</th>
      <td>Jessica Chastain</td>
      <td>Veronica Mars</td>
    </tr>
    <tr>
      <th>2393</th>
      <td>Jessica Chastain</td>
      <td>ER</td>
    </tr>
  </tbody>
</table>
<p>2394 rows × 2 columns</p>
</div>

Count how many shared actors/actress each movie/TV show has
```python
count_df = df[['movie_tv']][df['movie_tv'] != "Interstellar"] #delete the row with movie_tv name is Interstellar
count_df = count_df.apply(pd.value_counts) #count the number of shared actors/actress
count_df = count_df.reset_index()
count_df = count_df.rename(columns={'index': 'Movie/TV show', 'movie_tv': 'Number of shared actors/actress'})
```

Show the top 10 movies/TV shows
```python
count_df[:10]
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
      <th>Movie/TV show</th>
      <th>Number of shared actors/actress</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NCIS</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Criminal Minds</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Shameless</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Dark Knight Rises</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CSI: Crime Scene Investigation</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Law &amp; Order: Special Victims Unit</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Law &amp; Order</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Law &amp; Order: Criminal Intent</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>American Horror Story</td>
      <td>4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Revenge</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


Plot the movies/TV shows with more than 4 shared actors/actress
```python
from matplotlib import pyplot as plt

plot_df = count_df[count_df['Number of shared actors/actress'] >= 4]

plt.figure(figsize=(15,10))
plt.bar(plot_df['Movie/TV show'], plot_df['Number of shared actors/actress'], align='center', alpha=0.5)
plt.xticks(plot_df['Movie/TV show'],  rotation=45)
plt.ylabel("Number of shared actors/actress", size =20)
plt.title("Movies/TV shows recommendation based on number of shared actors/actress from $\it{Interstellar}$", size = 15)
```

![hw2_img1.png](/images/hw2_img1.png)