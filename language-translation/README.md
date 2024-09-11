# Languag
This is a FastAPI proxy that can be used to proxy requests to a generative AI API. It is a simple wrapper around the
generative AI API that allows for additional functionality such as:
* content rewrite
* content post rewrite
* title suggestion based on the given content
* title notification suggestion based on the given content
* notification description suggestion based on the given content
## Prerequisites
* poetry
* Python 3.9
* markdown-toc
* npm -> NodeJS package manager
* virtualenv

This README's TOC is managed by `markdown-toc` plugin that can be installed using `npm`
```shell
npm install -g markdown-toc