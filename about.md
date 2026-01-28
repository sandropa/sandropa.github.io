---
layout: article # Or 'article' if you prefer the TeXt styling for it and it fits
title: About Me
permalink: /
---

<!-- Add this image block -->
<img src="/assets/images/sandro.jpg" alt="Sandro Paradžik" style="float: right; margin: 0 0 1em 1em; width: 200px; border-radius: 50%;"> 
<!-- Adjust width, border-radius, and margins as needed -->
<!-- For float: left, use margin: 0 1em 1em 0; -->

Hello, my name is Sandro Paradžik. I am a Machine Learning intern at ModelCat, where we work on automating the creation of ML models for edge devices, mainly for computer vision. I live in Sarajevo and recently completed my Bachelor's in CS at the University of Sarajevo. In high school, I competed in mathematics and now I regularly give lectures to students preparing for competitions.

### Contact

*   [sandropa@hey.com](mailto:sandropa@hey.com){:target="_blank"}
*   [GitHub](https://github.com/sandropa){:target="_blank"}
*   [LinkedIn](https://www.linkedin.com/in/sandropa/){:target="_blank"}

### Favorite Books

<div class="book-cards-container">
  <div class="book-card">
    <img src="/assets/images/book_a_brief_history_of_intelligence.png" alt="Book Cover">
    <div class="book-info">
      <p class="book-title">A Brief History of Intelligence</p>
      <p class="book-author">Max Bennett</p>
    </div>
  </div>
  <div class="book-card">
    <img src="/assets/images/book_why_greatness_cannot_be_planned.png" alt="Book Cover">
    <div class="book-info">
      <p class="book-title">Why Greatness Cannot Be Planned</p>
      <p class="book-author">Kenneth O. Stanley, Joel Lehman</p>
    </div>
  </div>
  <div class="book-card">
    <img src="/assets/images/book_brave_new_world.png" alt="Book Cover">
    <div class="book-info">
      <p class="book-title">Brave New World</p>
      <p class="book-author">Aldous Huxley</p>
    </div>
  </div>
  <div class="book-card">
    <img src="/assets/images/book_a_thousand_brains.png" alt="Book Cover">
    <div class="book-info">
      <p class="book-title">A Thousand Brains</p>
      <p class="book-author">Jeff Hawkins</p>
    </div>
  </div>
</div>

### Favorite Movies

<div class="movie-cards-container">
  <div class="movie-card">
    <img src="/assets/images/movie_true_romance_2.png">
    <div class="movie-info">
      <p class="movie-title">True Romance</p>
      <p class="movie-director">Tony Scott</p>
    </div>
  </div>
  <div class="movie-card">
    <img src="/assets/images/movie_shutter_island.png" alt="Movie Cover">
    <div class="movie-info">
      <p class="movie-title">Shutter Island</p>
      <p class="movie-director">Martin Scorsese</p>
    </div>
  </div>
  <div class="movie-card">
    <img src="/assets/images/movie_dr_strangelove.png" alt="Movie Cover">
    <div class="movie-info">
      <p class="movie-title">Dr. Strangelove</p>
      <p class="movie-director">Stanley Kubrick</p>
    </div>
  </div>
  <div class="movie-card">
    <img src="/assets/images/movie_inception.png" alt="Movie Cover">
    <div class="movie-info">
      <p class="movie-title">Inception</p>
      <p class="movie-director">Christopher Nolan</p>
    </div>
  </div>
</div>

### Photographs

I like photography and believe art is very important. Here are some photographs I have taken over the years.

<div class="image-gallery">
  <figure>
    <img src="/assets/images/photographs/1_tb_bw_rain_small.jpg" alt="photo">
  </figure>
  <figure>
    <img src="/assets/images/photographs/2021_0505_photo1_small.jpeg" alt="photo">
  </figure>
  <figure>
    <img src="/assets/images/photographs/2021_0505_photo2_small.jpeg" alt="photo">
  </figure>
  <figure>
    <img src="/assets/images/photographs/foggy_small.jpg" alt="photo">
  </figure>
  <figure>
    <img src="/assets/images/photographs/mostar1_small.jpg" alt="photo">
  </figure>
  <figure>
    <img src="/assets/images/photographs/sarajevo_cat_1_small.jpg" alt="photo">
  </figure>
  <figure>
    <img src="/assets/images/photographs/skate1_small.jpg" alt="photo">
  </figure>
  <figure>
    <img src="/assets/images/photographs/sarajevo_shadow1.webp" alt="photo">
  </figure>
</div>

### Blog
coming soon...
<ul class="blog-list">
  {% for post in site.posts %}
    <li>
      <span class="post-date">{{ post.date | date: "%b %d, %Y" }}</span>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>