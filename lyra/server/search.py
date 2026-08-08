
import requests

try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

class WebSearchEngine:
    """
    Agentic Web Search Tool for fetching live supplemental facts, weather, news, sports,
    and external information requested or implied by the ambient user prompt.
    """

    def __init__(self, max_results: int = 4):
        self.max_results = max_results

    def search(self, query: str) -> list[dict[str, str]]:
        """
        Executes web search query and returns list of top result snippets.
        """
        if not query.strip():
            return []

        results = []

        if DDG_AVAILABLE:
            try:
                with DDGS() as ddgs:
                    ddg_gen = ddgs.text(query, max_results=self.max_results)
                    for r in ddg_gen:
                        results.append({
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "url": r.get("href", "")
                        })
                if results:
                    return results
            except Exception as e:
                print(f"[SearchEngine] DDGS error: {e}")

        # Fallback to DuckDuckGo HTML parsing or Wikipedia/Open API if needed
        try:
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            resp = requests.get(url, headers=headers, timeout=4)
            if resp.status_code == 200:
                # Basic extraction of text snippets
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, 'html.parser')
                snippets = soup.find_all('a', class_='result__snippet')
                titles = soup.find_all('a', class_='result__title')
                for i in range(min(len(snippets), self.max_results)):
                    results.append({
                        "title": titles[i].get_text(strip=True) if i < len(titles) else "Web Result",
                        "snippet": snippets[i].get_text(strip=True),
                        "url": titles[i].get('href', '') if i < len(titles) else ""
                    })
        except Exception as e:
            print(f"[SearchEngine] Fallback search error: {e}")

        return results

    def format_search_for_prompt(self, search_results: list[dict[str, str]]) -> str:
        """Formats search results for LLM prompt ingestion."""
        if not search_results:
            return "(No web search results found)"

        formatted = []
        for i, item in enumerate(search_results, 1):
            formatted.append(f"[{i}] {item['title']}\n    Snippet: {item['snippet']}\n    Source: {item['url']}")

        return "\n\n".join(formatted)
