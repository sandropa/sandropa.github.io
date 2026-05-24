---
layout: home
permalink: /photographs
---

<a href="{{ '/' | relative_url }}" class="home-link">Sandro Paradžik</a>

<div class="photos">
  {% assign photos = "foggy_small.jpg,1_tb_bw_rain_small.jpg,sarajevo_cat_1_small.jpg,skate1_small.jpg,2021_0505_photo1_small.jpeg,mostar1_small.jpg,2021_0505_photo2_small.jpeg,sarajevo_shadow1.webp" | split: "," %}
  {% for photo in photos %}
    <img src="{{ '/assets/images/photographs/' | append: photo | relative_url }}" alt="" loading="lazy">
  {% endfor %}
</div>
