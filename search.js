/* Search for blog.html.
 *
 * blog/search-index.json holds the title, tags and headings of every post. It
 * is fetched once, on the first sign the visitor means to search, so the page
 * itself stays as cheap to load as the rest of the site. With the index in
 * memory a query is a linear scan -- at a few thousand posts that is well
 * under a millisecond, which is not worth a real index structure. */

(function () {
  "use strict";

  var input = document.getElementById("q");
  var results = document.getElementById("search-results");
  var status = document.getElementById("search-status");
  var browse = document.getElementById("browse");
  if (!input || !results || !status || !browse) return;

  var MAX_RESULTS = 60;
  var posts = null;
  var loading = null;

  /* "giai thich" should find "Giải thích", and typing Vietnamese without a
     Vietnamese keyboard is the common case. Both the query and the indexed
     text go through this, so the two always meet in the middle. */
  function fold(text) {
    return text
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/đ/g, "d")
      .replace(/Đ/g, "d")
      .toLowerCase();
  }

  function load() {
    if (loading) return loading;
    // build.py stamps the index's content hash into this URL, so a rebuilt
    // index is a fresh fetch rather than whatever the browser already has.
    loading = fetch(input.dataset.index || "blog/search-index.json")
      .then(function (response) {
        if (!response.ok) throw new Error(response.status);
        return response.json();
      })
      .then(function (data) {
        posts = data.map(function (post) {
          return {
            title: post.t,
            slug: post.u,
            date: post.d,
            tags: post.g || [],
            headings: post.h || [],
            fTitle: fold(post.t),
            fTags: fold((post.g || []).join(" ")),
            fHeadings: (post.h || []).map(fold),
          };
        });
        return posts;
      });
    return loading;
  }

  /* A term counts as matching where it starts a word: "trans" finds
     "Transformers" but "former" does not, which keeps short queries from
     dragging in most of the archive. */
  function hasTerm(haystack, term) {
    var at = haystack.indexOf(term);
    while (at !== -1) {
      if (at === 0 || !/[a-z0-9]/.test(haystack.charAt(at - 1))) return true;
      at = haystack.indexOf(term, at + 1);
    }
    return false;
  }

  function score(post, terms) {
    var total = 0;
    for (var i = 0; i < terms.length; i++) {
      var term = terms[i];
      var points = 0;
      var matched = [];
      if (hasTerm(post.fTitle, term)) points += 8;
      if (hasTerm(post.fTags, term)) points += 4;
      for (var j = 0; j < post.fHeadings.length; j++) {
        if (hasTerm(post.fHeadings[j], term)) {
          points += 2;
          matched.push(j);
          break;
        }
      }
      // Every term has to land somewhere, so extra words narrow the search
      // rather than widening it.
      if (!points) return null;
      total += points;
      post.hit = post.hit || [];
      if (matched.length) post.hit.push(matched[0]);
    }
    return total;
  }

  function search(query) {
    var terms = fold(query).split(/\s+/).filter(Boolean);
    if (!terms.length) return [];
    var found = [];
    for (var i = 0; i < posts.length; i++) {
      posts[i].hit = [];
      var points = score(posts[i], terms);
      if (points !== null) found.push({ post: posts[i], score: points });
    }
    // Best match first, then newest, so equally relevant posts read in the
    // same order as every other listing on the site.
    found.sort(function (a, b) {
      return b.score - a.score || (a.post.date < b.post.date ? 1 : -1);
    });
    return found;
  }

  function text(tag, className, value) {
    var el = document.createElement(tag);
    if (className) el.className = className;
    el.textContent = value;
    return el;
  }

  function render(found, query) {
    results.textContent = "";
    for (var i = 0; i < Math.min(found.length, MAX_RESULTS); i++) {
      var post = found[i].post;
      var li = document.createElement("li");

      var link = document.createElement("a");
      link.href = "blog/" + post.slug + ".html";
      link.textContent = post.title;
      li.appendChild(link);
      li.appendChild(document.createTextNode(" "));
      li.appendChild(text("span", "date", "(" + post.date.replace(/-/g, "/") + ")"));

      // Say why the post matched when the title alone does not show it.
      if (post.hit.length && !hasTerm(post.fTitle, fold(query).split(/\s+/)[0])) {
        li.appendChild(document.createElement("br"));
        li.appendChild(text("span", "date hit", post.headings[post.hit[0]]));
      }
      results.appendChild(li);
    }

    var shown = Math.min(found.length, MAX_RESULTS);
    if (!found.length) {
      status.textContent = "No post matches " + JSON.stringify(query) + ".";
    } else if (found.length > shown) {
      status.textContent =
        "Showing " + shown + " of " + found.length + " matching posts.";
    } else {
      status.textContent =
        found.length === 1 ? "1 matching post." : found.length + " matching posts.";
    }
    status.hidden = false;
    results.hidden = false;
    browse.hidden = true;
  }

  function clear() {
    results.hidden = true;
    status.hidden = true;
    browse.hidden = false;
    results.textContent = "";
  }

  function run() {
    var query = input.value.trim();
    if (!query) return clear();
    load().then(
      function () {
        // A slow index can land after the box has been cleared or retyped.
        if (input.value.trim() !== query) return;
        render(search(query), query);
      },
      function () {
        status.textContent = "Search is unavailable — the index failed to load.";
        status.hidden = false;
        results.hidden = true;
        browse.hidden = false;
      }
    );
  }

  // Warm the index on the first hint of intent, so the first keystroke has it.
  input.addEventListener("focus", load, { once: true });
  input.addEventListener("input", run);
  input.addEventListener("search", run); // the type="search" clear button
  input.form &&
    input.form.addEventListener("submit", function (event) {
      event.preventDefault();
    });

  // ?q=... makes a search linkable.
  var initial = new URLSearchParams(location.search).get("q");
  if (initial) {
    input.value = initial;
    run();
  }
})();
