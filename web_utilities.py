import asyncio
import logging
import re
import requests

from bs4 import BeautifulSoup
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from markdownify import markdownify as md
from playwright.async_api import async_playwright
from typing import Any, AsyncIterator, Dict, List, Iterator, Optional, Sequence, Union


logger = logging.getLogger(__name__)

UNWANTED_SECTIONS = {
    "references",
    "external links",
    "further reading",
    "see also",
    "notes",
}


def build_metadata(soup: Any, url: str) -> dict:
    """Build metadata from BeautifulSoup output."""
    metadata = {"source": url}
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    if description := soup.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", "No description found.")
    if html := soup.find("html"):
        metadata["language"] = html.get("lang", "No language found.")
    return metadata


class MarkdownWebBaseLoader(WebBaseLoader):
    """
    A WebBaseLoader subclass that uses Playwright to render JS, then
    strips boilerplate and converts structured pieces to Markdown.
    """
    def __init__(
        self,
        web_path: Union[str, Sequence[str]] = "",
        header_template: Optional[dict] = None,
        verify_ssl: bool = True,
        proxies: Optional[dict] = None,
        continue_on_failure: bool = False,
        autoset_encoding: bool = True,
        encoding: Optional[str] = None,
        web_paths: Sequence[str] = (),
        requests_per_second: int = 2,
        default_parser: str = "html.parser",
        requests_kwargs: Optional[Dict[str, Any]] = None,
        raise_for_status: bool = False,
        bs_get_text_kwargs: Optional[Dict[str, Any]] = None,
        bs_kwargs: Optional[Dict[str, Any]] = None,
        session: Any = None,
        markdown_kwargs: Optional[Dict[str, Any]] = None,
        unwanted_css: Optional[List[str]] = None,
        unwanted_headings: Optional[List[str]] = None,
        render_wait: float = 1.0,
        *,
        show_progress: bool = True,
        trust_env: bool = False,
    ) -> None:
        """Initialize loader.

        Args:
            markdown_kwargs: Optional[Dict[str, Any]]: Arguments for markdownify.
            unwanted_css: Optional[List[str]]: CSS selectors to remove from the page.
            unwanted_headings: Optional[List[str]]: Headings to remove from the page.
            render_wait: float: Time to wait for JS rendering (default: 2.0 seconds).
        """
        super().__init__(
            web_path=web_path,
            header_template=header_template,
            verify_ssl=verify_ssl,
            proxies=proxies,
            continue_on_failure=continue_on_failure,
            autoset_encoding=autoset_encoding,
            encoding=encoding,
            web_paths=web_paths,
            requests_per_second=requests_per_second,
            default_parser=default_parser,
            requests_kwargs=requests_kwargs,
            raise_for_status=raise_for_status,
            bs_get_text_kwargs=bs_get_text_kwargs,
            bs_kwargs=bs_kwargs,
            session=session,
            show_progress=show_progress,
            trust_env=trust_env,
        )
        self.markdown_kwargs = markdown_kwargs or {
            "heading_style": "ATX",
            "bullets": "*+-",
            "strip": ["a", "span"],
            "table_infer_header": True
        }
        self.unwanted_css = unwanted_css or [
            ".toc", ".navbox", ".sidebar", ".advertisement", ".cookie-banner", ".vertical-navbox",
            ".hatnote", ".reflist", ".mw-references-wrap"
        ]
        self.unwanted_headings = [h.lower() for h in (unwanted_headings or UNWANTED_SECTIONS)]
        self.render_wait = render_wait

    @staticmethod
    def _should_render(html: str, soup: Any) -> bool:
        low_text = len(soup.get_text(strip=True)) < 100
        has_noscript = bool(soup.find("noscript"))
        cf_challenge = "just a moment" in html.lower() or "enable javascript" in html.lower()
        many_scripts = len(soup.find_all("script")) > 20
        return has_noscript or cf_challenge or low_text or many_scripts

    async def _fetch_with_playwright(self, url: str) -> str:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            # If you need cookies/auth, you can do:
            # await page.set_extra_http_headers(self.session.headers)
            await page.goto(url)
            await asyncio.sleep(self.render_wait)  # allow JS to finish
            content = await page.content()
            await browser.close()
            return content

    def _scrape(
            self,
            url: str,
            parser: Union[str, None] = None,
            bs_kwargs: Optional[dict] = None,
    ) -> Any:
        if parser is None:
            parser = "xml" if url.endswith(".xml") else self.default_parser
        self._check_parser(parser)

        resp = self.session.get(url, **self.requests_kwargs)
        if self.raise_for_status:
            resp.raise_for_status()
        if self.encoding is not None:
            resp.encoding = self.encoding
        elif self.autoset_encoding:
            resp.encoding = resp.apparent_encoding
        html = resp.text

        soup = BeautifulSoup(html, parser, **(bs_kwargs or {}))

        # If the html looks JS-heavy, re-render with Playwright
        if not url.endswith(".xml") and self._should_render(html, soup):
            try:
                rendered = asyncio.run(self._fetch_with_playwright(url))
                soup = BeautifulSoup(rendered, parser, **(bs_kwargs or {}))
            except Exception as e:
                logger.warning("Playwright rendering failed for %s: %s. Falling back to requests.", url, e)

        return soup

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Collapse runs of spaces, tabs, etc. down to single spaces—but skip
        inside fenced code blocks ```…``` or inline code `…`.
        """
        # Replace non-breaking and invisible spaces with regular spaces
        text = text.replace("\u00A0", " ")
        # Strip zero-width spaces:
        text = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", text)

        # Split out fenced code -> keep code blocks intact while normalizing other text
        parts = re.split(r'(```.*?```)', text, flags=re.S)
        for i, part in enumerate(parts):
            if not part.startswith("```"):
                # further split out inline code
                subparts = re.split(r'(`[^`\n]+`)', part)
                for j, sp in enumerate(subparts):
                    if not sp.startswith("`"):
                        # collapse whitespace, strip edges of each segment
                        subparts[j] = re.sub(r'[ \t\r\f\v]+', ' ', sp).strip()
                parts[i] = "".join(subparts)
        # Rejoin and ensure paragraphs are separated by a single blank line
        normalized = "\n\n".join(p for p in parts if p.strip() != "")
        return normalized

    def _convert_soup_to_text(self, soup: Any) -> str:
        # Strip scripts & styles
        for tag in soup(["script", "style"]):
            tag.decompose()
        # Drop blocks whose first heading matches unwanted
        for sec in soup.find_all(["section", "div", "aside"]):
            h = sec.find(["h1", "h2", "h3", "h4", "h5", "h6"])
            if h and any(h.get_text(strip=True).lower().startswith(u) for u in self.unwanted_headings):
                sec.decompose()
        # Drop by CSS selector
        for sel in self.unwanted_css:
            for el in soup.select(sel):
                el.decompose()
        # Isolate the main content container if present
        soup = soup.find("div", class_="mw-parser-output") or soup.find("main") or soup.find("article") or soup

        # Convert to Markdown text with markdownify
        markdown = md(str(soup), **self.markdown_kwargs)
        markdown = self.normalize_whitespace(markdown)
        return markdown

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path."""
        for path in self.web_paths:
            soup = self._scrape(path, bs_kwargs=self.bs_kwargs)
            text = self._convert_soup_to_text(soup)
            metadata = build_metadata(soup, path)
            yield Document(page_content=text, metadata=metadata)

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async lazy load text from the url(s) in web_path."""
        results = await self.ascrape_all(self.web_paths)
        for path, soup in zip(self.web_paths, results):
            text = self._convert_soup_to_text(soup)
            metadata = build_metadata(soup, path)
            yield Document(page_content=text, metadata=metadata)


def fetch_wikipedia_page(page_key: str, lang: str = "en") -> Dict[str, str]:
    """Fetches a Wikipedia page by its key and returns its content in Markdown format.

    Args:
        page_key (str): The unique key of the Wikipedia page.
        lang (str): The language code for the Wikipedia edition to fetch (default: "en").
    """
    page_key = page_key.replace(" ", "_")  # Ensure the page key is URL-safe
    page_url = f"https://api.wikimedia.org/core/v1/wikipedia/{lang}/page/{page_key}/html"
    visit_website_tool = MarkdownWebBaseLoader(page_url)
    markdown = visit_website_tool.load()[0].page_content
    return {
        "page_key": page_key,
        "markdown": markdown,
    }


def get_wikipedia_article(query: str, lang: str = "en") -> Dict[str, str]:
    """Searches and fetches a Wikipedia article for a given query and returns its content in Markdown format.

    Args:
        query (str): The search query.
        lang (str): The language code for the Wikipedia edition to search (default: "en").
    """
    headers = {
        'User-Agent': 'MyLLMAgent (llm_agent@example.com)'
    }

    search_url = f"https://api.wikimedia.org/core/v1/wikipedia/en/search/page"
    search_params = {'q': query, 'limit': 1}
    search_response = requests.get(search_url, headers=headers, params=search_params, timeout=15)

    if search_response.status_code != 200:
        raise Exception(f"Search error: {search_response.status_code} - {search_response.text}")

    results = search_response.json().get("pages", [])
    if not results:
        raise Exception(f"No results found for query: {query}")

    page = results[0]
    page_key = page["key"]

    return fetch_wikipedia_page(page_key, lang)


def parse_sections(markdown_text: str) -> Dict[str, Dict]:
    """
    Parses markdown into a nested dict:
    { section_title: {
         "full": full_section_md,
         "subsections": { sub_title: sub_md, ... }
      }, ... }
    """
    # First split top-level sections
    top_pat = re.compile(r"^##\s+(.*)$", re.MULTILINE)
    top_matches = list(top_pat.finditer(markdown_text))
    sections: Dict[str, Dict] = {}
    for i, m in enumerate(top_matches):
        sec_title = m.group(1).strip()
        start = m.start()
        end = top_matches[i+1].start() if i+1 < len(top_matches) else len(markdown_text)
        sec_md = markdown_text[start:end].strip()

        # Now split subsections within this block
        sub_pat = re.compile(r"^###\s+(.*)$", re.MULTILINE)
        subs: Dict[str, str] = {}
        sub_matches = list(sub_pat.finditer(sec_md))
        for j, sm in enumerate(sub_matches):
            sub_title = sm.group(1).strip()
            sub_start = sm.start()
            sub_end = sub_matches[j+1].start() if j+1 < len(sub_matches) else len(sec_md)
            subs[sub_title] = sec_md[sub_start:sub_end].strip()

        sections[sec_title] = {"full": sec_md, "subsections": subs}
    return sections
