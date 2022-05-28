---
layout: post
title: Homework 5
---
## Web Development

In this blog, I will build a web app that allows users to submit messages and view random submitted messages.
This is the repo for my app:[https://github.com/zrh99199/HW5](https://github.com/zrh99199/HW5)

### 0. Build app.py format
Import some useful modules into our app.py file. Our web will have one main page and two extended pages(submit and view). Here is how our app.py will look like:

```python
from flask import Flask, g, render_template, request
import sqlite3

app = Flask(__name__)

# The function will run on ('url')
@app.route('/')
def main():
    # add code

# The function will run on ('url/submit/')
@app.route('/submit/', methods=['POST', 'GET'])
def submit():
    # add code

# The function will run on ('url/view/')
@app.route('/view/')
def view(): 
    # add code
   
```

### 1. Build base template and style.css

The base template will define our web basic layout. The style.css will specify how the elements in our web will look like.

#### base.html
```python
<!DOCTYPE html>
<html lang="en">
  <head>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
  </head>

  <body>
    <header>
      <h1>A Simple Message Bank</h1>
    </header>

    <main>
      <nav>
        {% block nav %}{% endblock %}
      </nav>

      <section class="content">
        <header>
          {% block header %}{% endblock %}
        </header>
        {% block content %}{% endblock %}
      </section>
    </main>

    <footer>
      <hr>
      <small>
        &copy; Steve Zhu
      </small>
    </footer>
  </body>

</html>
```

style.css
```python
html {
    font-family: Arial (sans-serif);
    background: lightblue;
    padding: 1rem;
}

body {
    max-width: 900px;
    padding-top: 60px;
    margin: 0 auto;
}

h1 {
    color: red;
    margin: 1rem 0;
    text-align: center;
}

h2 {
    margin: 1rem 0;
    font-size: 23px;
}

a {
    color: CornflowerBlue;
    text-decoration: none;
}


nav {
    display:flex;
    list-style-type: none;
    margin:0;
    padding:0;
    overflow: hidden;
    background-color: lightyellow;
}

    nav ul {
        display: flex;
        list-style: none;
        margin: 0;
        padding: 0;
        width: 100%;
    }

        nav ul li a {
            display: block;
            padding: 0.5rem;
        }

        .active {
            background-color: red;
            color: blue;
        }

.content {
    padding: 0 1rem 1rem;
    background: lightyellow;
}

```

### 2. Build main page
The main page of our web are extended from base template and it will let user redirect to either submission page or message view page from the navigation bar.

```python
{% extends 'base.html' %}
{% block title %}Main{% endblock %}

{% block nav %}
<ul>
  <li><a href="{{ url_for('submit') }}">Submit a message</a></li>
  <li><a href="{{ url_for('view') }}">View messages</a></li>
</ul>
{% endblock %}

{% block header %}<h2>Welcome to Steve Zhu's message bank!</h2>{% endblock %}

```

In app.py, the main function will render main template

```python
# The function will run on ('url')
@app.route('/')
def main():
    return render_template('main.html')
```

### 3. Build submission page

Because we will upload the data into our database, we will need two functions in our app.py to help us get access to our database.
The first function is called get_message_db(). It will first check whether there is a data. If not, then connect to that database, ensuring that the connection is an attribute of g. Then, it will check whether a table called messages exists in message_db, and create it if not. In the end, it will return a connection to the database.

```python
def get_message_db():
    try:
        return g.message_db

    except:
        # create a sql database
        g.message_db = sqlite3.connect("messages_db.sqlite")

        # use cmd to create a table with three columns
        cmd = \
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            handle TEXT NOT NULL)
        """

        # execute cmd by cursor
        cursor = g.message_db.cursor()
        cursor.execute(cmd)

        return g.message_db
```

The second function is called insert_message(). It will first save user's input as vairables and upload them into the database.

```python
def insert_message():
    
    # save user's message input as variable 'message' and handle input as variable 'handle'
    message = request.form['message']
    handle = request.form['handle']

    # connect to the database and use cmd to save inputs
    conn = get_message_db()
    cmd = \
    f"""
    INSERT INTO messages (message, handle) 
    VALUES ('{message}', '{handle}')
    """

    cursor = conn.cursor()
    cursor.execute(cmd)

    conn.commit()
    conn.close()

    return message, handle
```

After we defined our two helper functions, we can build our submit template and submit function in app.py.

There are two methods in the submit page in app.py, POST and GET. Normally, the web is under GET method. But when user try to submit message in the website, the POST method will be active and allow user to enter message to our database.

```python
# The function will run on ('url/submit/'), the function will use POST and GET methods
@app.route('/submit/', methods=['POST', 'GET'])
def submit():

    # render the submit.html without any argument when there is no submitted form exists
    if request.method == 'GET':
        return render_template('submit.html')

    # else submit the message to the database
    else:
        try:
            message, handle = insert_message()
            return render_template('submit.html', submitted=True, message=message, handle=handle)

        except:
            return render_template('submit.html', error=True)
```

The submit page of our web are extended from base template too and it will allow user to submit a message and a handle(or name) into our data base. I also use class="active" in the link of submit in the nav bar to make the submit link to become different when we are already in the submit page
```python
{% extends 'base.html' %}
{% block title %}Submit{% endblock %}

{% block nav %}
<ul>
  <li><a href="{{ url_for('submit') }}" class="active">Submit a message</a></li>
  <li><a href="{{ url_for('view') }}">View messages</a></li>
</ul>
{% endblock %}

{% block header %}<h2>Submit a message</h2>{% endblock %}

{% block content %}
  <form method="post">
      <label for="message">Your message:</label><br>
      <textarea name="message" id="message"></textarea>
      <br><br>
      <label for="handle">Your name or handle:</label><br>
      <input type="text" name="handle" id="handle">
      <br><br>
      <input type="submit" value="Submit message">
  </form>

  {% if submitted %}
    <br>
    Congrulation! You made a submission! 
  {% endif %}

  {% if error %}
    <br>
    Error, please try again!
  {% endif %}
{% endblock %}
```


### 4. Build view page

First, we need a build a function called random_messages(n) in app.py. It will return a collection of n random messages from the message_db.

```python
def random_messages(n):

    # get n random message from our database
    conn = get_message_db()
    cmd = \
    f"""
    SELECT * FROM messages ORDER BY RANDOM() LIMIT {n}
    """
    cursor = conn.cursor()
    cursor.execute(cmd)

    # use fetchall to save the random message as variable 'result'
    result = cursor.fetchall()
    conn.close()

    return result
```

This is how view function in app.py will be.
```python
# The function will run on ('url/view/')
@app.route('/view/')
def view(): 
    return render_template('view.html', messages=random_messages(5))
```

The view page of our web is also extended from base template. And it will generate some random data from our database and output them.
```python
{% extends 'base.html' %}
{% block title %}View{% endblock %}

{% block nav %}
<ul>
  <li><a href="{{ url_for('submit') }}">Submit a message</a></li>
  <li><a href="{{ url_for('view') }}" class="active">View messages</a></li>
</ul>
{% endblock %}

{% block header %}<h2>Some Cool Messages</h2>{% endblock %}

{% block content %}
  {% for message in messages %}
    <br>
    </b>{{ message[1] }}
    <br>
    <i>- {{ message[2] }}</i>
    <br>
  {% endfor %}
{% endblock %}
```

### 5. Run web app

After finishing all the previous steps, we can now run our web app! We need to enter the following code in the terminal and in the right working direction(Where your app is).
```python
export FLASK_ENV=development
flask run
```

This is how the main page looks like:
![HW5-pic1.png](/images/HW5-pic1.png)

This is how the submit page looks like:
![HW5-pic2.png](/images/HW5-pic2.png)

After you submit your data, you can view them in the view page:
![HW5-pic3.png](/images/HW5-pic3.png)