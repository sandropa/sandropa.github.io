---
layout: home
permalink: /
---

<h1 class="site-name">Sandro Paradžik</h1>

<p class="bio">I love mathematics — probability, statistics, linear algebra, combinatorics — and I'm interested in AI and machine learning. I live in Sarajevo, where I also teach students preparing for math competitions.</p>

<ul class="links">
  <li><a href="mailto:sandropa@hey.com">email</a></li>
  <li><a href="https://github.com/sandropa">github</a></li>
  <li><a href="https://www.linkedin.com/in/sandropa">linkedin</a></li>
  <li><a href="{{ '/assets/pdf/cv_sandro.pdf' | relative_url }}">cv</a></li>
  <li><a href="https://world.hey.com/sandropa">blog</a></li>
</ul>

<ul class="posts">
  {% for post in site.posts %}
  <li>
    <time>{{ post.date | date: "%Y-%m-%d" }}</time>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </li>
  {% endfor %}
</ul>
