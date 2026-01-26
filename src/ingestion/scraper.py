"""
Documentation Scraper Module

Scrapes technical documentation and extracts:
1. Content chunks (sections, paragraphs, code blocks)
2. Structural relationships (parent-child sections, hyperlinks)

This builds the raw data needed for both ChromaDB and NetworkX graphs.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urljoin, urlparse
import time
import hashlib
from utils.logger import get_logger
from utils.custom_exception import CustomException 
import sys

logger = get_logger(__name__)

class DocumentNode:
    """Represents a documentation chunk with relationships."""
    def __init__(self, node_id:str, content:str, section_type:str, url:str, metadata: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.content = content
        self.section_type = section_type
        self.url = url
        self.metadata = metadata or {}
        self.links = [] #Outgoing hyperlinks to other nodes
    
    def add_link(self, target_url:str):
        """Add a hyperlink to another document."""
        self.links.append(target_url)
    
    def __repr__(self):
        return f"DocumentNode(id={self.node_id}, type={self.section_type}, links={len(self.links)})"

class DocumentationScraper:
    """
    Scrapes technical documentation and builds structured nodes.

    Supports:
    - Any HTML-based technical docs with consistent structure
    """

    def __init__(self, base_url:str, max_pages:int=50, delay:float=1.0):
        """
        Initialize the scraper.

        Args:
            base_url: Starting URL for documentation
            max_pages: Maximum number of pages to scrap (avoid overwhelming sites)
            delay: Delay between requests in seconds
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls = set()
        self.nodes = []

        # Parse base domain for filtering links
        self.base_domain = urlparse(base_url).netloc

        logger.info(f"Scraper initialized: {base_url}")

    def _generate_node_id(self, url:str, content:str) -> str:
        """Generate a unique node ID from URL and content hash."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"node_{url_hash}_{content_hash}"
    
    def _is_valid_url(self, url:str) -> bool:
        """Check if URL should be scraped (same domain, not already visited)."""
        parsed = urlparse(url)

        # Must be same domain
        if parsed.netloc != self.base_domain:
            return False
        
        # Skip non-HTML resources
        skip_extensions = ['.pdf', '.zip', '.tar', '.gz', '.jpg', '.png', '.svg']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        # Not already visited
        if url in self.visited_urls:
            return False
        
        return True 
    
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a single page."""
        try:
            logger.info(f"Fetching: {url}")

            headers = {
                'User-Agent': "Mozilla/5.0 (Cogito Educational Project)"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            return soup

        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def _extract_sections(self, soup: BeautifulSoup, url: str):
        """
        Extract content sections from the page.
        This creates nodes for:
            - Headers (h1, h2, h3)
            - Paragraphs
            - Code Blocks
            - Tables (optional)
        """
        nodes = []

        # Extract main content area (adjust selector based on docs site)
        main_content = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
        
        if not main_content:
            logger.warning(f"Could not find main content in {url}")
            main_content = soup.body
        
        if not main_content:
            return nodes
        
        # Process all relevant elements
        for element in main_content.find_all(['h1','h2','h3','p','pre','code']):
            content = element.get_text(strip=True)
            # Skip empty or very short content
            if len(content) < 18:
                continue
            # Determine section type
            if element.name in ['h1','h2','h3']:
                section_type = 'header'
            elif element.name in ['pre','code']:
                section_type = 'code'
            else:
                section_type = 'paragraph'
            # Generate node ID
            node_id = self._generate_node_id(url, content)
            # Create node
            node = DocumentNode(node_id=node_id,content=content,section_type=section_type,url=url,
                                metadata={'source':'Web_Scrape','page_url':url,'element_tag':element.name})

            # Extract hyperlinks from this element
            for link in element.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                node.add_link(full_url)
            
            nodes.append(node)
        logger.info(f"Extracted {len(nodes)} sections from {url}")
        return nodes
    
    def scrape(self, start_url: Optional[str]=None) -> List[DocumentNode]:
        """
        Scrape documentation starting from base_url.

        Args:
            start_url: Optional override for starting URL
        Returns:
            List of DocumentNode Objects
        """
        try:
            url = start_url or self.base_url
            to_visit = [url]

            while to_visit and len(self.visited_urls) < self.max_pages:
                current_url = to_visit.pop(0)

                if not self._is_valid_url(current_url):
                    continue

                self.visited_urls.add(current_url)

                # Fetch page
                soup = self._fetch_page(current_url)
                if not soup:
                    continue
                
                # Extract sections
                page_nodes = self._extract_sections(soup, current_url)
                self.nodes.extend(page_nodes)

                # Find new links to visit
                for link in soup.find_all('a', href=True):
                    full_url = urljoin(current_url, link['href'])
                    if self._is_valid_url(full_url) and len(to_visit) < self.max_pages:
                        to_visit.append(full_url)
                
                # Delay between requests
                time.sleep(self.delay)
                logger.info(f"Progress: {len(self.visited_urls)}/{self.max_pages} pages, {len(self.nodes)} nodes")
            
            logger.info(f"Scraping complete: {len(self.nodes)} nodes from {len(self.visited_urls)} pages")
            return self.nodes
        
        except Exception as e:
            raise CustomException(f"Scraping Failed: {e}",sys)
        
    def get_scrape_stats(self) -> Dict[str, Any]:
        """Get statistics about the scraping session."""
        section_types = {}
        for node in self.nodes:
            section_types[node.section_type] = section_types.get(node.section_type, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "pages_visited": len(self.visited_urls),
            "section_types": section_types,
            "base_url": self.base_url
        }