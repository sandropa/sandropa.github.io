---
layout: home
permalink: /
---

<div class="intro">
  <img class="portrait" src="{{ '/assets/images/profile.jpg' | relative_url }}" alt="Sandro Paradžik">
  <h1 class="site-name">Sandro Paradžik</h1>
  <p>Hello, my name is Sandro Paradžik. I am interested in AI/ML and Data Science, and more broadly, I love building things. In mathematics, I especially enjoy combinatorics, linear algebra, probability, and statistics.</p>
  <p>I studied Computer Science in Sarajevo. I've done internships in AI/ML, including a research internship at the Max Planck Institute in Tübingen. I also teach students preparing for math competitions, and I find that work deeply rewarding.</p>
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
<div class="gallery">
  <div class="gallery-pair">
    <img style="flex: 1" src="{{ '/assets/images/photographs/1_tb_bw_rain_small.jpg' | relative_url }}" alt="">
    <img style="flex: 1.56" src="{{ '/assets/images/photographs/foggy_small.jpg' | relative_url }}" alt="">
  </div>
  <div class="gallery-pair">
    <img style="flex: 1.78" src="{{ '/assets/images/photographs/skate1_small.jpg' | relative_url }}" alt="">
    <img style="flex: 1.33" src="{{ '/assets/images/photographs/sarajevo_shadow1.webp' | relative_url }}" alt="">
  </div>
  <div class="gallery-pair">
    <img style="flex: 1" src="{{ '/assets/images/photographs/2021_0505_photo1_small.jpeg' | relative_url }}" alt="">
    <img style="flex: 1.5" src="{{ '/assets/images/photographs/sarajevo_cat_1_small.jpg' | relative_url }}" alt="">
  </div>
  <div class="gallery-pair">
    <img style="flex: 1.33" src="{{ '/assets/images/photographs/mostar1_small.jpg' | relative_url }}" alt="">
    <img style="flex: 1.5" src="{{ '/assets/images/photographs/2021_0505_photo2_small.jpeg' | relative_url }}" alt="">
  </div>
</div>