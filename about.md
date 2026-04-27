---
layout: home
permalink: /
---

<section class="intro">
  <div>
    <p>Hello, my name is Sandro Paradžik. I love mathematics — especially probability theory, statistics, linear algebra, and combinatorics. I am also very interested in AI and machine learning, both in how these technologies are being applied to transform real-world workflows and industries, and in the research that drives them forward.</p>
    <p>I live in Sarajevo, where I also give lectures to students preparing for math competitions, something I've been doing since my own competing days in high school. I also keep a more personal blog on <a href="https://world.hey.com/sandropa">HEY World</a> — occasional notes and reflections.</p>
    <p>I am always happy to hear from people — feel free to reach out at <a href="mailto:sandropa@hey.com">sandropa@hey.com</a>.</p>
  </div>
  <img class="portrait" src="{{ '/assets/images/sandro.jpg' | relative_url }}" alt="Sandro Paradžik">
</section>

{% include writing.html %}

<h2 class="section">favorite books</h2>
<div class="grid-cards">
  <figure>
    <div class="frame"><img src="{{ '/assets/images/book_a_brief_history_of_intelligence.png' | relative_url }}" alt=""></div>
    <div class="meta"><div class="title">A Brief History of Intelligence</div><div class="by">Max Bennett</div></div>
  </figure>
  <figure>
    <div class="frame"><img src="{{ '/assets/images/book_why_greatness_cannot_be_planned.png' | relative_url }}" alt=""></div>
    <div class="meta"><div class="title">Why Greatness Cannot Be Planned</div><div class="by">Stanley &amp; Lehman</div></div>
  </figure>
  <figure>
    <div class="frame"><img src="{{ '/assets/images/book_brave_new_world.png' | relative_url }}" alt=""></div>
    <div class="meta"><div class="title">Brave New World</div><div class="by">Aldous Huxley</div></div>
  </figure>
  <figure>
    <div class="frame"><img src="{{ '/assets/images/book_a_thousand_brains.png' | relative_url }}" alt=""></div>
    <div class="meta"><div class="title">A Thousand Brains</div><div class="by">Jeff Hawkins</div></div>
  </figure>
</div>

<h2 class="section">favorite movies</h2>
<div class="grid-cards">
  <figure>
    <div class="frame"><img src="{{ '/assets/images/movie_true_romance_2.png' | relative_url }}" alt=""></div>
    <div class="meta"><div class="title">True Romance</div><div class="by">Tony Scott</div></div>
  </figure>
  <figure>
    <div class="frame"><img src="{{ '/assets/images/movie_shutter_island.png' | relative_url }}" alt=""></div>
    <div class="meta"><div class="title">Shutter Island</div><div class="by">Martin Scorsese</div></div>
  </figure>
  <figure>
    <div class="frame"><img src="{{ '/assets/images/movie_dr_strangelove.png' | relative_url }}" alt=""></div>
    <div class="meta"><div class="title">Dr. Strangelove</div><div class="by">Stanley Kubrick</div></div>
  </figure>
  <figure>
    <div class="frame"><img src="{{ '/assets/images/movie_inception.png' | relative_url }}" alt=""></div>
    <div class="meta"><div class="title">Inception</div><div class="by">Christopher Nolan</div></div>
  </figure>
</div>

<h2 class="section">photographs</h2>
<p>I like photography and believe art is very important. Here are some photographs I have taken over the years.</p>

<div class="photos">
  {% assign photos = "1_tb_bw_rain_small.jpg,2021_0505_photo1_small.jpeg,2021_0505_photo2_small.jpeg,foggy_small.jpg,mostar1_small.jpg,sarajevo_cat_1_small.jpg,skate1_small.jpg,sarajevo_shadow1.webp" | split: "," %}
  {% for photo in photos %}
    {% assign src = '/assets/images/photographs/' | append: photo | relative_url %}
    <figure><a href="{{ src }}" class="lb-link"><img src="{{ src }}" alt=""></a></figure>
  {% endfor %}
</div>
