---
layout: home
permalink: /
---

<div class="intro">
  <div>
    <h1 class="site-name">Sandro Paradžik</h1>
    <p class="bio">I love mathematics — probability, statistics, linear algebra, combinatorics — and I'm interested in AI and machine learning. I live in Sarajevo, where I also teach students preparing for math competitions.</p>
  </div>
  <img class="portrait" src="{{ '/assets/images/profile.jpg' | relative_url }}" alt="Sandro Paradžik">
</div>

<ul class="links">
  <li><a href="mailto:sandropa@hey.com">email</a></li>
  <li><a href="https://github.com/sandropa">github</a></li>
  <li><a href="https://www.linkedin.com/in/sandropa">linkedin</a></li>
  <li><a href="{{ '/assets/pdf/cv_sandro.pdf' | relative_url }}">cv</a></li>
  <li><a href="https://world.hey.com/sandropa">hey world</a></li>
</ul>

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