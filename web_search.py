import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0 Safari/537.36 GabeAI/0.1"
)
TOKEN_RE = re.compile(r"[a-z0-9']+")
LOW_QUALITY_TEXT = {
    "job",
    "jobs",
    "login",
    "log",
    "employer",
    "employers",
    "contributors",
    "advertise",
    "cookie",
}
DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
DDG_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>|'
    r'<div[^>]+class="result__snippet"[^>]*>(?P<snippet2>.*?)</div>',
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    page_text: str = ""
    score: float = 0.0

    def to_dict(self):
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "pageText": self.page_text[:1200],
            "score": round(self.score, 3),
        }


class VisibleTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip_depth = 0
        self.parts = []
        self.active_tag = ""

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        self.active_tag = tag
        if tag in {"script", "style", "noscript", "svg", "canvas"}:
            self.skip_depth += 1
        if tag in {"p", "h1", "h2", "h3", "li", "article", "section", "title"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg", "canvas"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"p", "h1", "h2", "h3", "li", "article", "section", "title"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if self.skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(text)

    def get_text(self):
        text = html.unescape(" ".join(self.parts))
        return " ".join(text.split())


class SearchClient:
    def __init__(self):
        pass

    def status(self):
        return {
            "provider": "DuckDuckGo HTML",
            "requiresApiKey": False,
            "codeVersion": "no-api-search-0.3",
            "searchFile": __file__,
        }

    def search_and_read(self, query, provider="auto", top_k=5):
        results, provider_used = self.search(query, provider=provider, top_k=top_k)
        question_terms = set(TOKEN_RE.findall(query.lower()))
        enriched = []

        for result in results[:top_k]:
            page_text = self.fetch_page_text(result.url)
            page_snippet = self.pick_snippet(page_text, question_terms)
            snippet = self.choose_snippet(result.snippet, page_snippet)
            score = self.score_result(query, result.title, result.url, snippet, page_text)
            enriched.append(
                SearchResult(
                    title=result.title,
                    url=result.url,
                    snippet=snippet,
                    page_text=page_text[:5000],
                    score=score,
                )
            )

        enriched.sort(key=lambda item: item.score, reverse=True)
        return enriched, provider_used

    def choose_snippet(self, search_snippet, page_snippet):
        if not page_snippet:
            return search_snippet or ""
        if not search_snippet:
            return page_snippet
        search_score = self.snippet_quality(search_snippet)
        page_score = self.snippet_quality(page_snippet)
        return page_snippet if page_score > search_score + 1.0 else search_snippet

    def snippet_quality(self, snippet):
        text = (snippet or "").lower()
        words = TOKEN_RE.findall(text)
        score = 0.0
        if 80 <= len(snippet) <= 360:
            score += 0.5
        if re.search(r"\bis (a|an|the|used|built|designed)\b", snippet, re.IGNORECASE):
            score += 1.0
        if any(term in text for term in ["open-source", "open source", "framework", "library", "documentation"]):
            score += 0.7
        if self.looks_like_navigation(words):
            score -= 2.0
        if any(term in text for term in ["registered trademark", "privacy policy", "frequently asked questions"]):
            score -= 1.5
        return score

    def search(self, query, provider="auto", top_k=5):
        ddg_results = self.search_duckduckgo(query, top_k=top_k)
        return ddg_results, "DuckDuckGo HTML"

    def search_duckduckgo(self, query, top_k=5):
        params = urllib.parse.urlencode({"q": query, "kl": "us-en"})
        data = self.read_url("https://html.duckduckgo.com/html/?" + params, limit=700000)
        titles = DDG_RESULT_RE.findall(data)
        snippets = []
        for match in DDG_SNIPPET_RE.finditer(data):
            snippets.append(match.group("snippet") or match.group("snippet2") or "")

        results = []
        for idx, (href, title_html) in enumerate(titles):
            url = self.clean_duckduckgo_url(html.unescape(href))
            title = self.clean_html(title_html)
            snippet = self.clean_html(snippets[idx]) if idx < len(snippets) else ""
            if url and title:
                results.append(SearchResult(title=title, url=url, snippet=snippet))
            if len(results) >= top_k:
                break
        return results

    def clean_duckduckgo_url(self, url):
        parsed = urllib.parse.urlparse(url)
        if parsed.path.startswith("/l/"):
            query = urllib.parse.parse_qs(parsed.query)
            if query.get("uddg"):
                return query["uddg"][0]
        if url.startswith("//"):
            return "https:" + url
        return url

    def fetch_page_text(self, url):
        if "google.com/search" in url:
            return ""
        try:
            data = self.read_url(url, limit=900000)
        except Exception:
            return ""
        parser = VisibleTextParser()
        try:
            parser.feed(data)
        except Exception:
            return ""
        return parser.get_text()

    def read_url(self, url, limit=500000):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=10) as response:
            content_type = response.headers.get("Content-Type", "")
            charset = "utf-8"
            match = re.search(r"charset=([^;\s]+)", content_type, re.IGNORECASE)
            if match:
                charset = match.group(1).strip()
            data = response.read(limit)
        return data.decode(charset, errors="replace")

    def pick_snippet(self, page_text, question_terms):
        if not page_text:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", page_text)
        ranked = []
        for index, sentence in enumerate(sentences):
            clean = " ".join(sentence.split())
            words = TOKEN_RE.findall(clean.lower())
            if 35 <= len(clean) <= 320 and not self.looks_like_navigation(words):
                score = len(question_terms.intersection(words))
                if re.search(r"\bis (a|an|the|used|built|designed)\b", clean, re.IGNORECASE):
                    score += 0.7
                if score:
                    ranked.append((score, -index, clean))
        ranked.sort(key=lambda row: row[0], reverse=True)
        return ranked[0][2] if ranked else page_text[:260]

    def score_result(self, query, title, url, snippet, page_text):
        terms = set(TOKEN_RE.findall(query.lower()))
        url_text = title + " " + snippet
        haystack = " ".join([url_text or "", page_text[:2000] or ""]).lower()
        words = TOKEN_RE.findall(haystack)
        if not terms or not words:
            return 0.0
        overlap = len(terms.intersection(words))
        density = sum(1 for word in words if word in terms) / max(1, len(words))
        return overlap + density + self.url_quality_bonus(title, url, snippet)

    def url_quality_bonus(self, title, url, snippet):
        text = " ".join([title or "", snippet or ""]).lower()
        host = urllib.parse.urlparse(url or "").netloc.lower()
        bonus = 0.0
        if host.endswith("pytorch.org") or host.endswith("python.org"):
            bonus += 2.0
        if host.endswith("wikipedia.org"):
            bonus += 1.0
        if host.endswith("github.com"):
            bonus += 0.4
        if "official" in text or "documentation" in text or "tutorial" in text:
            bonus += 0.6
        if "wikipedia" in text:
            bonus += 0.4
        if self.looks_like_navigation(TOKEN_RE.findall(text)):
            bonus -= 1.3
        return bonus

    def looks_like_navigation(self, words):
        if not words:
            return False
        low_quality_hits = sum(1 for word in words if word in LOW_QUALITY_TEXT)
        return low_quality_hits >= 3

    def clean_html(self, value):
        value = re.sub(r"<.*?>", " ", value or "")
        return " ".join(html.unescape(value).split())
