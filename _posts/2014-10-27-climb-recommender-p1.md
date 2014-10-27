---
layout: post
title: "Building a Climbing Route Recommender - Part 1"
excerpt: "Acquiring route rating data"
tags: [climbing, machine learning, recommenders]
date: 2014-10-27
---

Content and product recommenders are all the rage currently. Services such as Netflix, Spotify, and Amazon have used techniques such as collaborative filtering to great success.  This series of posts will cover acquiring ratings data, and building a distributed collaborative filtering algorithm in Apache Spark.

For my personal curiosities, I've decided to use ratings from the popular rock climbing site [The Mountain Project](http://www.mountainproject.com). The Mountain Project contains millions of ratings for tens of thousands of climbs from all over the world, but there isn't a nice API for accessing the data.  What we will have to do is to crawl the climbing areas and sub-areas to collect the ratings data. Fortunately, [Scrapy](http://scrapy.org/) makes it quite easy for us to implement a crawler.  So we're going to create a Scrapy crawler that follows the links and parses the xpath values of interest.

First thing is to start a new Scrapy project:


```
scrapy startproject MPspider
```

The we need to modify the `items.py` file to capture the fields we want:


{% highlight python %}
 import scrapy.item

 class MpspiderItem(scrapy.Item):
    # define the fields for your item here like:
    climbName = scrapy.Field()
    climbID = scrapy.Field()
    climbGrade = scrapy.Field()
    userID = scrapy.Field()
    userStars = scrapy.Field()
    pass
{% endhighlight %}


Then we need to define our spider, and put it in the `spiders` directory.  The code will look something like this:

{% highlight python %}
import scrapy
from MPspider.items import MpspiderItem
from re import search

class MySpider(scrapy.Spider):
	name = "mountainproject"
	allowed_domains = ["mountainproject.com"]
	start_urls = ["page_to_start_crawling"]

	def parse_ratings(self, response):
		
		users = response.xpath("xpath_to_users_and_ratings")

		for user in users:

			# get item attributes with xpath
			
			yield item
{% endhighlight %}
