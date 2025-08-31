---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---
# Blog and Personal Website

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