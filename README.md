# mikyx-1.github.io

Personal site. Static HTML/CSS with a small Python build step for blog posts.
Styled after [joschu.net](http://joschu.net/index.html): Georgia serif, `#444`
on `#fefefe`, 60em max width, Wikipedia-blue links, dark top nav.

## Layout

```
index.html        bio + links (two-column: text left, photo right)  — hand-written
cv.html           CV                                                — hand-written
blog.html         blog index                                        — GENERATED
blog/*.html       rendered posts                                    — GENERATED
posts/*.md        post sources, with YAML front matter              — edit these
style.css         the whole stylesheet
build.py          renders posts/ -> blog/ and rewrites blog.html
images/           photos, figures, favicons
.nojekyll         tells GitHub Pages to serve these files as-is, not run Jekyll
```

Anything marked GENERATED is overwritten by `build.py` — edit `posts/*.md`, not
`blog/*.html`.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Writing a post

Add a markdown file to `posts/` named `YYYY-MM-DD-slug.md`:

```markdown
---
title: 'Your Title'
date: 2026-08-14
tags:
  - computer vision
---

Body text here.
```

Then rebuild:

```
.venv/bin/python build.py
```

It writes `blog/slug.html` and regenerates the index, newest first. Posts dated
in the future are skipped, so they work as drafts.

Heading ids use kramdown's slug rule, matching what Jekyll produced, so the
hand-written tables of contents inside the migrated posts still resolve.

## Preview locally

```
python3 -m http.server 8000
```

Then open http://localhost:8000. Use a server rather than `file://` so the
root-relative links (`/index.html`) resolve.

## Deploy

GitHub Pages serves the repo root. Commit and push to the default branch; the
`.nojekyll` file stops Pages from trying to build it as a Jekyll site.

Remember to commit the generated `blog/` and `blog.html` — Pages serves them
directly and does not run `build.py`.
