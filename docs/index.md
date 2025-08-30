---
layout: default
---

# Welcome to Alex Kozlov's Website

This is a minimal Jekyll site for testing the Docker development environment.

## About

Personal website and blog hosted on GitHub Pages.

## Latest Posts

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <span class="date">{{ post.date | date: "%B %-d, %Y" }}</span>
    </li>
  {% endfor %}
</ul>
