#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# http://0.0.0.0:8000/
from __future__ import unicode_literals

AUTHOR = 'Joseph Willard'
SITENAME = "Joseph Willard's Blog"
SITEURL = ''

PATH = 'content'

TIMEZONE = 'America/Detroit'

DEFAULT_LANG = 'en'

THEME = '/home/joseph/git_projects/pelican-themes/bootstrap2'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Twiecki\'s', 'https://twiecki.io/'),
         ('PyMC3', 'https://docs.pymc.io/'),)

# Social widget
SOCIAL = (('github', 'https://github.com/josephwillard'),
          ('linkedin', 'https://www.linkedin.com/in/joseph-willard-5040a4117/'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
