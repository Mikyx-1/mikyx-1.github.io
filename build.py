#!/usr/bin/env python3
"""Render posts/YYYY/*.md into blog/*.html and regenerate the index pages.

Writes, all of it overwritten on every run:

    blog/<slug>.html            one page per post
    blog/search-index.json      title, tags and headings, for search.js
    blog.html                   search box over the full list of posts

Usage:  .venv/bin/python build.py
"""

import hashlib
import json
import re
import unicodedata
from datetime import date
from pathlib import Path

import markdown

ROOT = Path(__file__).parent
POSTS_DIR = ROOT / "posts"
BLOG_DIR = ROOT / "blog"
SITE_TITLE = "Le Hoang Viet's Homepage"
EMAIL = "lehoangviet2k@gmail.com"
SITE_URL = "https://mikyx-1.github.io"
AUTHOR = "Le Hoang Viet"
AUTHOR_BIBTEX = "Le, Hoang Viet"  # family name first, as BibTeX expects

POST_TEMPLATE = """<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} &mdash; Le Hoang Viet</title>
<link rel="icon" href="../images/favicon.ico" sizes="any">
<link rel="stylesheet" href="../style.css">
{math_head}</head>
<body>
<div class="topnav">
  <a href="/index.html">{site_title}</a>
</div>

<h2>{title}</h2>
<p class="date">{date}{tags}</p>

{body}

<hr>

<h2>Citation</h2>

<p>Cited as:</p>

<blockquote>{citation}</blockquote>

<p>Or the BibTeX entry:</p>

<pre><code>{bibtex}</code></pre>

<hr>
<p><a href="../blog.html">&larr; Back to the blog index</a>.
Send feedback by <a href="mailto:{email}">email</a>.</p>
{math_foot}</body>
</html>
"""

# Only the posts that actually contain formulas pull KaTeX in; everything else
# stays a plain static page with no scripts at all.
KATEX_VERSION = "0.18.4"
MATH_HEAD = """<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/katex@{v}/dist/katex.min.css"
  integrity="sha384-u1zONI5gPXUx0UKI62c75/zww972y0v2rSK5ZYlVdS6xEuWDeZWUI66v6t1gvlXJ"
  crossorigin="anonymous">
<script defer
  src="https://cdn.jsdelivr.net/npm/katex@{v}/dist/katex.min.js"
  integrity="sha384-ykMNcWQhhTUb0YV9SPpPUFURHZ+tWmubkakGBP+OgNK/UXdO2gtzglWx0Rj9hnO3"
  crossorigin="anonymous"></script>
""".format(v=KATEX_VERSION)

# The formulas were pulled out before markdown ran, so each one sits in its own
# element already -- no need for auto-render to guess where the delimiters are.
MATH_FOOT = """<script defer>
window.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".math").forEach(function (el) {
    katex.render(el.textContent, el, {
      displayMode: el.classList.contains("math-display"),
      throwOnError: false,
    });
  });
});
</script>
"""

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blog Index &mdash; Le Hoang Viet</title>
<link rel="icon" href="images/favicon.ico" sizes="any">
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="topnav">
  <a href="/index.html">{site_title}</a>
</div>

<h2>Blog Index</h2>

<input type="search" id="q" class="search" autocomplete="off" spellcheck="false"
  data-index="{index_url}"
  placeholder="Search {count} posts by title, tag, or section">
<p id="search-status" class="date" hidden></p>
<ul id="search-results" hidden></ul>

<div id="browse">
<ul>
{items}
</ul>
</div>

<p>Send feedback by <a href="mailto:{email}">email</a>.</p>
<script defer src="search.js"></script>
</body>
</html>
"""


def parse_front_matter(text):
    """Pull `key: value` pairs and `- item` lists from a leading --- block."""
    if not text.startswith("---"):
        return {}, text
    _, fm, body = text.split("---", 2)
    meta, key = {}, None
    for line in fm.strip().splitlines():
        if line.startswith("  - ") or line.startswith("- "):
            if key:
                meta.setdefault(key, []).append(line.split("-", 1)[1].strip())
        elif ":" in line:
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip().strip("'\"")
            if value:
                meta[key] = value
    return meta, body.lstrip()


def kramdown_slugify(value, separator):
    """Reproduce Jekyll/kramdown heading ids.

    The posts carry hand-written tables of contents whose links were generated
    against kramdown, so the ids have to match its rule exactly: lowercase,
    drop anything that is not a letter, digit, space, or hyphen, then swap
    spaces for hyphens. Note it does NOT collapse the runs of separators left
    behind by stripped punctuation -- "Tips & Best Practices" becomes
    "tips--best-practices", with two hyphens.
    """
    value = unicodedata.normalize("NFC", value).lower()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE).replace("_", "")
    return value.replace(" ", separator)


LIST_MARKER = re.compile(r"^([-*+]|\d+\.)\s+\S")


def separate_lists(text):
    """Insert the blank line python-markdown needs before a list.

    Kramdown lets a list interrupt a paragraph directly:

        Trong do:
        - `H, W` -- chieu cao

    python-markdown does not, and would render the whole thing as one
    paragraph with stray hyphens. The posts were written against kramdown, so
    restore the blank line rather than reformat every source file.

    Only a list opening straight after an unindented paragraph line is
    touched; continuation lines inside an existing list are left alone, so
    tight lists stay tight.
    """
    out, in_fence, prev = [], False, ""
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif (
            not in_fence
            and LIST_MARKER.match(line)
            and prev.strip()
            and not prev.startswith((" ", "\t"))
            and not LIST_MARKER.match(prev)
        ):
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


# Ordered alternation: code and raw HTML are matched first, so a `$` inside a
# fenced block, a backtick span or an attribute -- one of the image URLs is
# `...?$pjpeg$&wid=768` -- is consumed there and never read as a formula.
MATH_TOKEN = re.compile(
    r"(?P<fence>^```.*?^```)"
    r"|(?P<code>`[^`\n]*`)"
    r"|(?P<tag><[a-zA-Z/!][^>]*>)"
    r"|(?P<display>\$\$.+?\$\$)"
    r"|(?P<inline>\$(?![\s$])[^$\n]*?(?<![\s\\])\$)",
    re.DOTALL | re.MULTILINE,
)

# Uppercase letters and digits only: markdown leaves it completely alone.
MATH_MARK = "MATHPLACEHOLDER{}ENDMATH"


def extract_math(text):
    """Swap every formula out for a placeholder before markdown runs.

    python-markdown would otherwise chew through the TeX -- `x_i ... x_j`
    becomes an <em>, backslashes get dropped -- so the formulas are lifted out
    first and put back as KaTeX-ready elements once the markdown is HTML.
    """
    formulas = []

    def stash(match):
        kind = match.lastgroup
        if kind in ("fence", "code", "tag"):
            return match.group(0)
        raw = match.group(0)
        if kind == "display":
            tex = raw[2:-2]
            # The posts also use $$ mid-sentence -- "the series of $$f(x)$$
            # around $$x = a$$" -- where display mode would break the line in
            # two. Only a $$ that has its own line stays a display formula.
            before = text.rfind("\n", 0, match.start()) + 1
            after = text.find("\n", match.end())
            trailing = text[match.end() : after if after != -1 else len(text)]
            if text[before : match.start()].strip() or trailing.strip():
                kind = "inline"
        else:
            tex = raw[1:-1]
        formulas.append((kind, tex.strip()))
        return MATH_MARK.format(len(formulas) - 1)

    return MATH_TOKEN.sub(stash, text), formulas


def restore_math(html, formulas):
    """Put the formulas back as elements the KaTeX snippet can render."""
    for index, (kind, tex) in enumerate(formulas):
        mark = MATH_MARK.format(index)
        # KaTeX reads the TeX from textContent, so it has to survive as text.
        text = tex.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if kind == "display":
            # A display formula sits alone in its paragraph; unwrap it so the
            # block-level KaTeX output is not nested inside a <p>.
            html = html.replace(
                f"<p>{mark}</p>", f'<div class="math math-display">{text}</div>'
            )
            html = html.replace(mark, f'<span class="math math-display">{text}</span>')
        else:
            html = html.replace(mark, f'<span class="math">{text}</span>')
    return html


def is_vietnamese(text):
    """Detect Vietnamese so the post gets the right lang attribute."""
    return bool(re.search(r"[ăâđêôơưĂÂĐÊÔƠƯ]|[ạảấầẩẫậắằẳẵặẹẻẽếềểễệ]", text))


HEADING_LINE = re.compile(r"^#{2,6}\s+(.+?)\s*#*$", re.MULTILINE)

# Several posts break their sections down with a bold run-in line rather than a
# real heading -- "**Bước 2**: Tính gradient và orientation" -- and those name
# the specific technique a reader is most likely to search for. A line only
# counts as one if it is short: past that it is emphasised prose, not a label.
RUN_IN_LINE = re.compile(r"^\*\*.+$", re.MULTILINE)
RUN_IN_MAX = 80


def clean_heading(raw):
    """Reduce a heading to the words in it, dropping the markdown around them."""
    text = re.sub(r"\$[^$]*\$|`[^`]*`", " ", raw)  # formulas and code spans
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links
    text = re.sub(r"[*_~]", "", text)  # bold and italic
    text = re.sub(r"^#{1,6}\s*", "", text.strip())
    text = re.sub(r"^\d+(\.\d+)*\.?\s*", "", text)  # leading section numbers
    return " ".join(text.split())


def collect_headings(text):
    """Pull the section headings out, as plain words, for the search index.

    Headings carry most of what a post is about -- "Wasserstein distance",
    "Kantorovich-Rubinstein duality" -- so indexing them finds far more than
    titles alone while staying small enough to ship as one file. Fenced code is
    skipped: every `#` line inside these posts is a Python comment.
    """
    prose = re.sub(r"^```.*?^```", "", text, flags=re.DOTALL | re.MULTILINE)

    found = [clean_heading(raw) for raw in HEADING_LINE.findall(prose)]
    found += [
        heading
        for heading in (clean_heading(raw) for raw in RUN_IN_LINE.findall(prose))
        if len(heading) <= RUN_IN_MAX
    ]

    # A post repeats labels like "Code" under every step; one is enough.
    headings, seen = [], set()
    for heading in found:
        if heading and heading.lower() not in seen:
            seen.add(heading.lower())
            headings.append(heading)
    return headings


def escape(text):
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


MONTHS = (
    "January February March April May June July "
    "August September October November December"
).split()

BIBTEX_TEMPLATE = """@misc{{{key},
  title   = {{{title}}},
  author  = {{{author}}},
  journal = {{{journal}}},
  year    = {{{year}}},
  month   = {{{month}}},
  url     = {{{url}}}
}}"""


def citation_for(title, post_date, slug):
    """The two forms a reader is likely to want: prose, and a BibTeX entry.

    Returns both already escaped for HTML, since they go straight into the
    page. A title carrying `{` or `}` would otherwise unbalance the BibTeX
    braces, so those are dropped rather than escaped -- they are not something
    a post title needs.
    """
    year, month = post_date[:4], MONTHS[int(post_date[5:7]) - 1]
    url = f"{SITE_URL}/blog/{slug}.html"

    prose = f'{AUTHOR}. "{title}". {SITE_TITLE}, {month} {year}. {url}'

    # A BibTeX key wants to be one word: leviet2024wgan.
    key = re.sub(r"[^a-z0-9]", "", AUTHOR.split()[0].lower() + year + slug)
    bibtex = BIBTEX_TEMPLATE.format(
        key=key,
        title=title.replace("{", "").replace("}", ""),
        author=AUTHOR_BIBTEX,
        journal=SITE_TITLE,
        year=year,
        month=month,
        url=url,
    )
    return escape(prose), escape(bibtex)


def render(md_path):
    meta, body = parse_front_matter(md_path.read_text(encoding="utf-8"))
    headings = collect_headings(body)

    # Posts were written for Jekyll, where images resolve from the site root.
    body = body.replace("](/images/", "](../images/")
    body = body.replace('src="/images/', 'src="../images/')
    body = separate_lists(body)
    body, formulas = extract_math(body)

    html = markdown.markdown(
        body,
        extensions=[
            "fenced_code",
            "codehilite",
            "tables",
            "attr_list",
            "sane_lists",
            "toc",
        ],
        extension_configs={
            "codehilite": {"guess_lang": False},
            "toc": {"slugify": kramdown_slugify},
        },
    )

    html = restore_math(html, formulas)

    # Wide tables scroll in their own box instead of stretching the page.
    html = html.replace("<table>", '<div class="table-wrap"><table>')
    html = html.replace("</table>", "</table></div>")

    tags = meta.get("tags", [])
    tag_text = f" &middot; {', '.join(tags)}" if tags else ""
    post_date = str(meta.get("date", ""))
    title = meta.get("title", md_path.stem)

    slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", md_path.stem)
    citation, bibtex = citation_for(title, post_date, slug)

    out_path = BLOG_DIR / f"{slug}.html"
    out_path.write_text(
        POST_TEMPLATE.format(
            lang="vi" if is_vietnamese(body) else "en",
            title=title,
            site_title=SITE_TITLE,
            date=post_date,
            tags=tag_text,
            body=html,
            citation=citation,
            bibtex=bibtex,
            email=EMAIL,
            math_head=MATH_HEAD if formulas else "",
            math_foot=MATH_FOOT if formulas else "",
        ),
        encoding="utf-8",
    )
    return {
        "title": title,
        "date": post_date,
        "slug": slug,
        "href": f"/blog/{slug}.html",
        "tags": tags,
        "headings": headings,
    }


def list_items(posts, prefix="/blog/"):
    """The one <li> shape every listing on the site uses."""
    return "\n".join(
        f'  <li><a href="{prefix}{p["slug"]}.html">{escape(p["title"])}</a> '
        f'<span class="date">({p["date"].replace("-", "/")})</span></li>'
        for p in posts
    )





def write_search_index(posts):
    """Everything search.js matches on, in one file it fetches once.

    Short keys and no whitespace: at a thousand posts the difference between
    this and a pretty-printed dump is most of a megabyte.
    """
    payload = [
        {
            "t": p["title"],
            "u": p["slug"],
            "d": p["date"],
            "g": p["tags"],
            "h": p["headings"],
        }
        for p in posts
    ]
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path = BLOG_DIR / "search-index.json"
    path.write_text(body, encoding="utf-8")

    # The filename never changes, so a browser holding yesterday's copy would
    # keep missing today's posts. Stamping the content hash into the URL that
    # blog.html asks for makes a changed index a different fetch.
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    return f"blog/search-index.json?v={digest}", len(body.encode("utf-8"))


def main():
    BLOG_DIR.mkdir(parents=True, exist_ok=True)

    sources = sorted(POSTS_DIR.glob("**/*.md"))
    seen = {}
    for path in sources:
        slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", path.stem)
        if slug in seen:
            raise SystemExit(f"{path} and {seen[slug]} both render to {slug}.html")
        seen[slug] = path

    posts = [render(p) for p in sources]

    # Newest first, and drop anything dated in the future (drafts).
    today = date.today().isoformat()
    posts = [p for p in posts if p["date"] <= today]
    posts.sort(key=lambda p: (p["date"], p["title"]), reverse=True)

    index_url, index_bytes = write_search_index(posts)

    (ROOT / "blog.html").write_text(
        INDEX_TEMPLATE.format(
            site_title=SITE_TITLE,
            count=len(posts),
            index_url=index_url,
            items=list_items(posts),
            email=EMAIL,
        ),
        encoding="utf-8",
    )

    index_kb = (ROOT / "blog.html").stat().st_size / 1024
    print(
        f"Built {len(posts)} post(s), blog.html {index_kb:.0f} KB, "
        f"search index {index_bytes / 1024:.0f} KB"
    )


if __name__ == "__main__":
    main()
