#!/usr/bin/env python3
"""Render posts/*.md into blog/*.html and regenerate blog.html.

Usage:  .venv/bin/python build.py
"""

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

POST_TEMPLATE = """<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} &mdash; Le Hoang Viet</title>
<link rel="icon" href="../images/favicon.ico" sizes="any">
<link rel="stylesheet" href="../style.css">
</head>
<body>
<div class="topnav">
  <a href="/index.html">{site_title}</a>
</div>

<h2>{title}</h2>
<p class="date">{date}{tags}</p>

{body}

<hr>
<p><a href="../blog.html">&larr; Back to the blog index</a>.
Send feedback by <a href="mailto:{email}">email</a>.</p>
</body>
</html>
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

<ul>
{items}
</ul>

<p>Send feedback by <a href="mailto:{email}">email</a>.</p>
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


def is_vietnamese(text):
    """Detect Vietnamese so the post gets the right lang attribute."""
    return bool(re.search(r"[ăâđêôơưĂÂĐÊÔƠƯ]|[ạảấầẩẫậắằẳẵặẹẻẽếềểễệ]", text))


def render(md_path):
    meta, body = parse_front_matter(md_path.read_text(encoding="utf-8"))

    # Posts were written for Jekyll, where images resolve from the site root.
    body = body.replace("](/images/", "](../images/")

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
            # slugify_unicode keeps Vietnamese diacritics in heading ids, so the
            # hand-written table-of-contents links in the posts still resolve.
            "toc": {"slugify": kramdown_slugify},
        },
    )

    # Wide tables scroll in their own box instead of stretching the page.
    html = html.replace("<table>", '<div class="table-wrap"><table>')
    html = html.replace("</table>", "</table></div>")

    tags = meta.get("tags", [])
    tag_text = f" &middot; {', '.join(tags)}" if tags else ""
    post_date = str(meta.get("date", ""))
    title = meta.get("title", md_path.stem)

    slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", md_path.stem)
    out_path = BLOG_DIR / f"{slug}.html"
    out_path.write_text(
        POST_TEMPLATE.format(
            lang="vi" if is_vietnamese(body) else "en",
            title=title,
            site_title=SITE_TITLE,
            date=post_date,
            tags=tag_text,
            body=html,
            email=EMAIL,
        ),
        encoding="utf-8",
    )
    return {"title": title, "date": post_date, "href": f"/blog/{slug}.html"}


def main():
    BLOG_DIR.mkdir(exist_ok=True)
    posts = [render(p) for p in sorted(POSTS_DIR.glob("*.md"))]

    # Newest first, and drop anything dated in the future (drafts).
    today = date.today().isoformat()
    posts = [p for p in posts if p["date"] <= today]
    posts.sort(key=lambda p: p["date"], reverse=True)

    items = "\n".join(
        f'  <li><a href="{p["href"]}">{p["title"]}</a> '
        f'<span class="date">({p["date"].replace("-", "/")})</span></li>'
        for p in posts
    )
    (ROOT / "blog.html").write_text(
        INDEX_TEMPLATE.format(site_title=SITE_TITLE, items=items, email=EMAIL),
        encoding="utf-8",
    )
    print(f"Built {len(posts)} post(s) into blog/ and regenerated blog.html")


if __name__ == "__main__":
    main()
