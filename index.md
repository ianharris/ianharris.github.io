---
title: "Ian Harris"
---

![Profile]({{ "/assets/profile.jpg" }}){:style="width:150px; height=150px;float:left;margin:10px;"}I am a Software Engineer currently working in the area of Data and Machine Learning Engineering. I aim to create a number of tutorials and articles on the topics of Data Science and Machine Learning. The content will be aimed at people new to the field and will hopefully serve as a useful learning tool.

There are only a few tutorials/articles here so far but I hope to add them soon. If you want to keep up to date, check out the RSS feed link below.

<div style="clear: both;"></div>
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

