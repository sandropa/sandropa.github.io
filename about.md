---
layout: home
permalink: /
---

<div class="intro">
  <img class="portrait" src="{{ '/assets/images/profile.jpg' | relative_url }}" alt="Sandro Paradžik">
  <h1 class="site-name">Sandro Paradžik</h1>
  <p>Hello, my name is Sandro Paradžik. I am interested in AI/ML and Data Science, and more broadly, I love building things. In mathematics, I especially enjoy combinatorics, linear algebra, probability, and statistics.</p>
  <p>I studied Computer Science in Sarajevo. I've done internships in AI/ML, including a research internship at the Max Planck Institute in Tübingen.</p>
  <p>For the past 4 years I've had the privilege of teaching at the Math School for Gifted Students, working with an incredible group of students preparing for math competitions. Our team recently achieved Bosnia and Herzegovina's best-ever results at the International Mathematical Olympiad, and being even a small part of that has been incredibly rewarding.</p>
  <p>Here I share thoughts on math, AI, and tech. You can also find some of my photography on this website.</p>
  <p>I am always happy to hear from people. Feel free to reach out at <a href="mailto:sandropa@hey.com">sandropa@hey.com</a>.</p>
</div>

<ul class="links">
  <li><a href="mailto:sandropa@hey.com">email</a></li>
  <li><a href="https://github.com/sandropa">github</a></li>
  <li><a href="https://www.linkedin.com/in/sandropa">linkedin</a></li>
  <li><a href="{{ '/assets/pdf/cv_sandro.pdf' | relative_url }}">cv</a></li>
  <li><a href="https://world.hey.com/sandropa">hey world</a></li>
</ul>

<p class="section-label">writing</p>

<ul class="posts">
  {% for post in site.posts %}
  <li>
    <time>{{ post.date | date: "%Y-%m-%d" }}</time>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </li>
  {% endfor %}
</ul>

<p class="section-label">photographs</p>
<div class="photo-strip">
  <div class="photo-strip-inner">
    {% assign photos = "foggy_small.jpg,1_tb_bw_rain_small.jpg,sarajevo_cat_1_small.jpg,skate1_small.jpg,2021_0505_photo1_small.jpeg,mostar1_small.jpg,2021_0505_photo2_small.jpeg,sarajevo_shadow1.webp" | split: "," %}
    {% for photo in photos %}
      <img src="{{ '/assets/images/photographs/' | append: photo | relative_url }}" alt="">
    {% endfor %}
  </div>
</div>