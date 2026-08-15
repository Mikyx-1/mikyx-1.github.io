# mikyx-1.github.io

Personal site. Static HTML/CSS with a small Python build step for blog posts.

Black on white, no color beyond the one link blue. Typography and page
furniture follow [jondeaton.github.io](https://jondeaton.github.io/): Optima
(humanist sans), 800px measure, `line-height: 1.5`, a right-aligned pill nav
over a dashed rule, portrait floated right at 200px, and code in Lucida
Console. Syntax highlighting is bold/italic only — no color.

Links take their `#0645ad` from [joschu.net](http://joschu.net/), and a visited
link keeps it instead of turning purple — on an index of a few hundred posts the
browser default paints half the page a second colour to report reading history,
which is not something the page is trying to say.

## Layout

```
index.html               bio + links (text left, photo right)   — hand-written
cv/Resume.pdf            the CV, linked from index.html         — drop in a new one
posts/YYYY/*.md          post sources, with YAML front matter   — edit these
blog.html                search box over every post             — GENERATED
blog/*.html              rendered posts                         — GENERATED
blog/search-index.json   what search.js matches on              — GENERATED
search.js                the search box on blog.html
style.css                the whole stylesheet
build.py                 renders posts/ -> blog/ and the index pages
images/                  photos, favicons
images/posts/            figures for one post, a directory per post
.nojekyll                serve these files as-is, do not run Jekyll
```

Anything marked GENERATED is overwritten by `build.py` — edit `posts/**/*.md`,
not `blog/`.

`posts/2024/` also holds `03/` and `05/`, two hand-written redirect stubs that
keep the old academicpages URLs working. They are not sources; `build.py` only
looks at `*.md`.

## How it scales

Sources sit in one directory per year. The rendered post keeps a flat URL
(`posts/2024/2024-05-18-wgan.md` → `/blog/wgan.html`), so moving a file between
year directories never changes a published link, and `build.py` refuses to run
if two sources would render to the same filename.

`blog.html` lists every post and lets the search box narrow it. A long list
costs less than it looks: measured on a synthetic 1000-post corpus, `blog.html`
is 153 KB of HTML that gzips to 13 KB, which is what Pages actually sends. The
search index is 484 KB, 34 KB gzipped. A full rebuild takes 12 seconds and a
query 0.2–0.5 ms.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Writing a post

Add a markdown file to `posts/<year>/` named `YYYY-MM-DD-slug.md`:

```markdown
---
title: 'Your Title'
date: 2026-08-14
tags:
  - Computer Vision
---

Body text here.
```

Then rebuild:

```
.venv/bin/python build.py
```

It writes `blog/slug.html` and regenerates the index, newest first. Posts dated
in the future are skipped, so they work as drafts.

Tags do not get their own pages; they show under the post title and are one of
the things search matches on.

## Citation

Every post ends with a Citation section — a line to quote and a BibTeX entry —
built from the title, date, and slug. Nothing to write per post.

The author name, site URL, and the `Le, Hoang Viet` form BibTeX wants are the
constants at the top of `build.py`. The entry key is the surname, year, and
slug run together (`le2024wgan`), which is unique because the slug is.

Heading ids use kramdown's slug rule, matching what Jekyll produced, so the
hand-written tables of contents inside the migrated posts still resolve.

## Formulas

Write TeX between `$…$` for a formula that sits in the line, and `$$…$$` on a
line of its own for one that gets its own block. Both are lifted out before
markdown runs, so underscores and backslashes survive, and `$` inside code, a
backtick span, or an HTML attribute is left alone.

`build.py` renders them with [KaTeX](https://katex.org/), pulled from a CDN and
only on the posts that actually contain a formula — every other page stays
script-free.

## Search

The box on `blog.html` matches titles, tags, and section headings — including
the bold run-in lines (`**Bước 2**: Tính gradient…`) the posts use as
sub-headings. Not body prose: at a thousand posts that is ~12 MB, against
~500 KB for this.

`blog/search-index.json` is fetched once, when the box is first focused, and
searched in memory afterwards. Accents are folded on both sides, so `giai
thich` finds `Giải thích`. Every word in the query has to match somewhere, so
adding words narrows the result. `blog.html?q=...` is a linkable search.

With JavaScript off the box does nothing and the full list below it, which is
plain HTML, still reaches every post.

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
