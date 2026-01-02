---
type: chain
name: post_writer
default: true
sequence:
- url_fetcher
- social_media
---
Chain processes requests through a series of agents in sequence, the output of each agent is passed to the next.
