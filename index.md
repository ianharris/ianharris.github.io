---
title: "Ian Harris"
---

![terminal-code]({{ "/assets/terminal-code.png" }}){:style="width=680px; height=330px; margin-bottom: 10px;"}

Below is a list of articles/tutorials on the topics of data science and data engineering as well as some more general software engineering posts. There are only a few tutorials/articles here so far but I hope to add them soon. If you want to keep up to date, check out the RSS feed link below.

If you want to know more about me, check out my LinkedIn profile with the link below or head over to this site's [about]({{ "/about.html" }}) page.

<a href="https://twitter.com/_ianharris?ref_src=twsrc%5Etfw"><img src="/assets/twitter.png" width="21" height="21"/></a>
<a href="https://www.linkedin.com/in/ian-harris-a9954652"><img src="/assets/linkedin.png" width="30" height="21"/><a/>
<a href="https://www.iharris.net/feed.xml"><img src="/assets/rss.png" width="21" height="21"/><a/>

{% for category in site.categories %}
**{{ category[0] }}**
<ul>{% for post in category[1] reversed %}
    {% unless post.url contains "appendix" %}
        <li>
            <a href="{{ post.url }}">
                {{ post.title }}
            </a>
        </li>
    {% endunless %}
    {% endfor %}
</ul>
**{{ category[0] }} Appendices**
<ul>
{% for post in category[1] reversed %}
{% if post.url contains "appendix" %}
  <li><a href="{{ post.url }}">{{ post.title }}</a></li>
{% endif %}
{% endfor %}
</ul>

{% endfor %}

