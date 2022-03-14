---
layout: post
title: Activity - Getting Started with Jekyll
---

In this post, we'll get set up with [Jekyll](https://jekyllrb.com/). Jekyll is a *static site converter*, which you can use to turn plaintext documents into attractive webpages. 

## Pre-Work: Install Jekyll

You should have already installed Jekyll when completing the activity in [this post]({{ site.baseurl }}/software).

## 1. Fork the Git Repo

Your first step is to fork the [GitHub repository](https://github.com/HarlinLee/HarlinLee.github.io) for this blog. Although this repository contains noticeably more files than the one we [practiced with](https://github.com/PIC16B/git-practice), the process of forking is almost exactly the same. Just click the "Fork" button on the top right of the GitHub page. 

After you fork the repository, go to Settings and change the repository name to 

```
[your GitHub username].github.io
```

For example, my username is `HarlinLee`, and so I would rename the repository to `HarlinLee.github.io`.

## 2. Publish on GitHub Pages

Now, go Settings again, and scroll down until you see the GitHub Pages section. Publish your blog! Don't modify any of the other settings. 

Wait a few minutes, and then navigate to 

```
https://[your username].github.io/
```

If you see a webpage there, congrats! Your blog is up and running. At the moment, it's just a copy of the template, so it's not personalized in any way. 

## 3. Call the blog by your name

Still on GitHub, find the file `_config.yml`. Click the pencil icon to edit the file. Change the `name` and `description` fields. Anything is fine! You're encouraged but not required to use your real name. 

In a few moments, your website will update with the new information. Now it's yours! 


## 4. Clone it

Your next step is to clone your blog. Make sure that you clone *your* blog (the fork that you just created) and not the original template. When you *clone* a repository, you download a local copy that you can modify. You'll then be able to *push* changes back to the version on GitHub. 

1. Click the big green Code download button. 
2. Choose "Open with GitHub Desktop."
3. Wait a moment for the files to download. 
4. You'll also need to select a location for the files. Choose a location that you'll be able to remember. 

## 5. Look around

Open up the repository that you just cloned in a file explorer, and take a look around. It should look something like this: 

```bash
[username].github.io
├── _includes/
├── _layouts/
├── _posts/
├── _sass/
├── _site/
├── images/
├── notebooks/
├── 404.md
├── CNAME
├── LICENSE
├── README.md
├── _config.yml
├── about.md
├── index.html
└── style.scss
```

You won't need to touch most of these files, but we'll soon take a quick tour. 

## 6. Serve Your Blog Locally

Jekyll is a utility that transforms *markdown* files, like the ones in this repository, to valid HTML files. In order to see a visual representation of your website locally, you need to *serve* the blog. 

1. Open a terminal. If you are using VS Code to edit your files, you can open a terminal inside the VS Code app itself using Terminal --> New Terminal. Or in GitHub Desktop, go to Repository -> Open in Terminal. 
2. Are you in the directory containing your blog? If not, navigate there using the `cd` command. 
3. In the terminal, type the command `jekyll serve --livereload`. You'll see a few printed messages in the terminal. 
4. Look for the *server address*, which will likely resemble `http://127.0.0.1:4000/`. Paste this address into your browser. You should see your blog! 

This copy isn't online; it's running directly on your computer. This allows you to (a) preview changes more rapidly than you can by editing on GitHub and (b) work on your blog using a text editor, which is usually more comfortable than the file modification interface on GitHub. Changes that you make to your blog files will be quickly and automatically reflected in the locally served version of your blog.

## 7. Make a Post

*Starting with this step, it is a good idea to commit each time you make a significant addition to your site.*

Create a file called `2022-03-31-test-post.md` in the `_posts` directory. Open the file in a text editor, and add the following text at the top of the post: 

```m
---
layout: post
title: My First Post!! 
---
```

Next, add some text. Any text will do: 

```
I'm an awesome PIC16B student and I am running Jekyll! 
```

Now, in your browser, navigate to your homepage (http://127.0.0.1:4000/). You should see a new blog post with title "My First Post" appear on the local version of your site. Click into it and observe that the text you added is now there. 

## 8. Edit and Push!

Try to add different components to your post as shown in [this page]({{ site.baseurl }}/composing).

Once you've made all these additions to your test post, publish the result. To do so, make sure you have committed all your changes in GitHub Desktop, including any files you may have added. Once you've done so, push! In a few minutes, you should see your new post on your website. 


