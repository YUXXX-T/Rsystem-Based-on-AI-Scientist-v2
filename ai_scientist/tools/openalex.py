import requests
import time
import warnings
from typing import Dict, List, Optional

import backoff

from ai_scientist.tools.base_tool import BaseTool


def on_backoff(details: Dict) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


class OpenAlexSearchTool(BaseTool):
    """
    Drop-in replacement for SemanticScholarSearchTool using OpenAlex API.
    Free, no API key required, generous rate limits (100k requests/day with polite pool).
    """

    def __init__(
        self,
        name: str = "SearchSemanticScholar",  # Keep same name so LLM prompts don't need changing
        description: str = (
            "Search for relevant literature using academic search. "
            "Provide a search query to find relevant papers."
        ),
        max_results: int = 10,
        contact_email: str = None,
    ):
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query to find relevant papers.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        # OpenAlex polite pool: include email for 100k/day rate limit (vs 10k/day without)
        self.contact_email = contact_email

    def use_tool(self, query: str) -> Optional[str]:
        try:
            papers = self.search_for_papers(query)
        except Exception as e:
            print(f"OpenAlex search failed after retries: {e}")
            return "Literature search unavailable. Proceeding without search results."
        if papers:
            return self.format_papers(papers)
        else:
            return "No papers found."

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
        on_backoff=on_backoff,
        max_tries=3,
    )
    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        if not query:
            return None

        params = {
            "search": query,
            "per_page": self.max_results,
            "select": "title,authorships,primary_location,publication_year,abstract_inverted_index,cited_by_count",
        }
        if self.contact_email:
            params["mailto"] = self.contact_email

        time.sleep(0.5)  # Be polite even with generous limits

        rsp = requests.get(
            "https://api.openalex.org/works",
            params=params,
        )
        print(f"Response Status Code: {rsp.status_code}")
        if rsp.status_code != 200:
            print(f"Response Content: {rsp.text[:500]}")
        rsp.raise_for_status()

        results = rsp.json()
        papers = results.get("results", [])
        if not papers:
            return None

        # Normalize to a common format
        normalized = []
        for paper in papers:
            authors = [
                authorship.get("author", {}).get("display_name", "Unknown")
                for authorship in paper.get("authorships", [])
            ]
            # Reconstruct abstract from inverted index
            abstract = self._reconstruct_abstract(
                paper.get("abstract_inverted_index")
            )
            venue = ""
            primary = paper.get("primary_location") or {}
            source = primary.get("source") or {}
            venue = source.get("display_name", "Unknown Venue")

            year = paper.get("publication_year", "Unknown Year")
            title = paper.get("title", "Unknown Title")

            # Generate BibTeX entry for compatibility with writeup modules
            first_author_last = authors[0].split()[-1] if authors else "unknown"
            cite_key = f"{first_author_last.lower()}{year}"
            bibtex = (
                f"@article{{{cite_key},\n"
                f"  title={{{title}}},\n"
                f"  author={{{' and '.join(authors)}}},\n"
                f"  journal={{{venue}}},\n"
                f"  year={{{year}}}\n"
                f"}}"
            )

            normalized.append(
                {
                    "title": title,
                    "authors": [{"name": a} for a in authors],
                    "venue": venue,
                    "year": year,
                    "abstract": abstract,
                    "citationCount": paper.get("cited_by_count", 0),
                    "citationStyles": {"bibtex": bibtex},
                }
            )

        # Sort by citation count descending
        normalized.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
        return normalized

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[Dict]) -> str:
        """Reconstruct abstract from OpenAlex's inverted index format."""
        if not inverted_index:
            return "No abstract available."
        # inverted_index: {"word": [position1, position2, ...], ...}
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)

    def format_papers(self, papers: List[Dict]) -> str:
        paper_strings = []
        for i, paper in enumerate(papers):
            authors = ", ".join(
                [author.get("name", "Unknown") for author in paper.get("authors", [])]
            )
            paper_strings.append(
                f"""{i + 1}: {paper.get("title", "Unknown Title")}. {authors}. {paper.get("venue", "Unknown Venue")}, {paper.get("year", "Unknown Year")}.
Number of citations: {paper.get("citationCount", "N/A")}
Abstract: {paper.get("abstract", "No abstract available.")}"""
            )
        return "\n\n".join(paper_strings)


# ── Standalone function (drop-in replacement for semantic_scholar.search_for_papers) ──

_tool_instance = OpenAlexSearchTool(max_results=10)


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
    on_backoff=on_backoff,
    max_tries=3,
)
def search_for_papers(query, result_limit=10):
    """Standalone search function compatible with semantic_scholar.search_for_papers interface."""
    if not query:
        return None
    _tool_instance.max_results = result_limit
    return _tool_instance.search_for_papers(query)

