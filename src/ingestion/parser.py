"""
Document Parser Module

Handles document parsing and chunking strategies for local files.
Supports:
1. Markdown (.md)
2. reStructuredText (.rst)
3. HTML files (.html)
4. Plain text (.txt)
5. PDF files (.pdf)

Chunking Strategies:
- Fixed-size chunks with overlap
- Semantic chunking (by headers/sections)
- Sliding window chunking

This complements the scraper.py for offline/local document ingestion.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
import markdown
from utils.logger import get_logger
from utils.custom_exception import CustomException
import sys

# PDF support (optional)
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    
logger = get_logger(__name__)


@dataclass
class ParsedChunk:
    """Represents a parsed and chunked document section."""
    chunk_id: str
    content: str
    chunk_type: str  # 'header', 'paragraph', 'code', 'mixed'
    source_file: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"ParsedChunk(id={self.chunk_id}, type={self.chunk_type}, len={len(self.content)})"


class DocumentParser:
    """
    Multi-format document parser with configurable chunking strategies.
    
    Supports parsing local files (Markdown, RST, HTML, TXT, PDF) and
    applying various chunking strategies for optimal RAG retrieval.
    """
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.md', '.markdown', '.rst', '.html', '.htm', '.txt', '.pdf'}
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50,
        chunking_strategy: Literal['fixed', 'semantic', 'sliding'] = 'semantic'
    ):
        """
        Initialize the document parser.
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size (skip smaller chunks)
            chunking_strategy: 'fixed', 'semantic', or 'sliding'
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.chunking_strategy = chunking_strategy
        
        logger.info(f"DocumentParser initialized: strategy={chunking_strategy}, "
                   f"chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def _generate_chunk_id(self, source: str, content: str, index: int) -> str:
        """Generate a unique chunk ID."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        return f"chunk_{source_hash}_{content_hash}_{index:04d}"
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type from extension."""
        ext = file_path.suffix.lower()
        if ext in {'.md', '.markdown'}:
            return 'markdown'
        elif ext == '.rst':
            return 'rst'
        elif ext in {'.html', '.htm'}:
            return 'html'
        elif ext == '.txt':
            return 'text'
        elif ext == '.pdf':
            return 'pdf'
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _parse_markdown(self, content: str) -> str:
        """Convert Markdown to plain text, preserving structure."""
        # Convert to HTML first, then extract text
        html = markdown.markdown(content, extensions=['fenced_code', 'tables'])
        soup = BeautifulSoup(html, 'lxml')
        return soup.get_text(separator='\n\n')
    
    def _parse_rst(self, content: str) -> str:
        """Parse reStructuredText to plain text."""
        # Basic RST parsing - remove directives and formatting
        # Remove RST directives like .. code-block::
        content = re.sub(r'\.\. \w+::[^\n]*\n', '\n', content)
        # Remove reference targets
        content = re.sub(r'\.\. _[^\n]+:\s*\n', '', content)
        # Remove inline markup but keep text
        content = re.sub(r'`([^`]+)`_', r'\1', content)  # Links
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)  # Bold
        content = re.sub(r'\*([^*]+)\*', r'\1', content)  # Italic
        content = re.sub(r'``([^`]+)``', r'\1', content)  # Inline code
        return content
    
    def _parse_pdf(self, file_path: Path) -> str:
        """Parse PDF to plain text."""
        if not PDF_SUPPORT:
            raise ImportError("PyPDF2 is not installed. Install with: pip install PyPDF2")
        
        text_parts = []
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            logger.warning(f"Error reading PDF {file_path}: {e}")
            raise
        
        return '\n\n'.join(text_parts)
    
    def _parse_html(self, content: str) -> str:
        """Parse HTML to plain text."""
        soup = BeautifulSoup(content, 'lxml')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get text from main content areas
        main = soup.find('article') or soup.find('main') or soup.find('div', class_='content') or soup.body
        
        if main:
            return main.get_text(separator='\n\n')
        return soup.get_text(separator='\n\n')
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by cleaning whitespace and empty lines."""
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines).strip()
    
    def _chunk_fixed(self, text: str, source: str) -> List[ParsedChunk]:
        """
        Fixed-size chunking with overlap.
        Simple but effective for uniform content.
        """
        chunks = []
        start = 0
        index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind('\n\n', start, end)
                if para_break > start + self.min_chunk_size:
                    end = para_break
                else:
                    # Look for sentence break
                    sentence_break = max(
                        text.rfind('. ', start, end),
                        text.rfind('.\n', start, end),
                        text.rfind('? ', start, end),
                        text.rfind('! ', start, end)
                    )
                    if sentence_break > start + self.min_chunk_size:
                        end = sentence_break + 1
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunk = ParsedChunk(
                    chunk_id=self._generate_chunk_id(source, chunk_text, index),
                    content=chunk_text,
                    chunk_type='mixed',
                    source_file=source,
                    metadata={
                        'strategy': 'fixed',
                        'char_start': start,
                        'char_end': end
                    }
                )
                chunks.append(chunk)
                index += 1
            
            start = end - self.chunk_overlap
            if start <= 0 and index > 0:
                break
        
        return chunks
    
    def _chunk_semantic(self, text: str, source: str) -> List[ParsedChunk]:
        """
        Semantic chunking based on document structure.
        Splits at headers and preserves logical sections.
        """
        chunks = []
        
        # Split by headers (Markdown-style or plain text headers)
        # Pattern matches: ## Header, === underline, --- underline
        header_pattern = r'(?:^|\n)(#{1,6}\s+[^\n]+|[^\n]+\n[=\-]{3,})'
        
        sections = re.split(header_pattern, text)
        
        current_section = ""
        current_header = ""
        index = 0
        
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            
            # Check if this is a header
            is_header = (
                section.startswith('#') or 
                (i + 1 < len(sections) and re.match(r'^[=\-]{3,}$', sections[i + 1].strip() if sections[i + 1] else ''))
            )
            
            if is_header:
                # Save previous section if exists
                if current_section and len(current_section) >= self.min_chunk_size:
                    chunk = ParsedChunk(
                        chunk_id=self._generate_chunk_id(source, current_section, index),
                        content=current_section,
                        chunk_type='paragraph' if not current_header else 'mixed',
                        source_file=source,
                        metadata={
                            'strategy': 'semantic',
                            'header': current_header
                        }
                    )
                    chunks.append(chunk)
                    index += 1
                
                current_header = section.lstrip('#').strip()
                current_section = section + "\n\n"
            else:
                current_section += section + "\n\n"
                
                # If section gets too large, split it
                if len(current_section) > self.chunk_size * 2:
                    sub_chunks = self._chunk_fixed(current_section, source)
                    for sub_chunk in sub_chunks:
                        sub_chunk.metadata['header'] = current_header
                        sub_chunk.metadata['strategy'] = 'semantic_overflow'
                    chunks.extend(sub_chunks)
                    index += len(sub_chunks)
                    current_section = ""
        
        # Don't forget the last section
        if current_section and len(current_section) >= self.min_chunk_size:
            chunk = ParsedChunk(
                chunk_id=self._generate_chunk_id(source, current_section, index),
                content=current_section.strip(),
                chunk_type='paragraph',
                source_file=source,
                metadata={
                    'strategy': 'semantic',
                    'header': current_header
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_sliding(self, text: str, source: str) -> List[ParsedChunk]:
        """
        Sliding window chunking.
        Creates overlapping chunks for better context coverage.
        """
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        index = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = ParsedChunk(
                        chunk_id=self._generate_chunk_id(source, chunk_text, index),
                        content=chunk_text,
                        chunk_type='mixed',
                        source_file=source,
                        metadata={
                            'strategy': 'sliding',
                            'sentence_count': len(current_chunk)
                        }
                    )
                    chunks.append(chunk)
                    index += 1
                
                # Keep overlap (last few sentences)
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) < self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_len
        
        # Last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = ParsedChunk(
                    chunk_id=self._generate_chunk_id(source, chunk_text, index),
                    content=chunk_text,
                    chunk_type='mixed',
                    source_file=source,
                    metadata={
                        'strategy': 'sliding',
                        'sentence_count': len(current_chunk)
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def parse_file(self, file_path: str) -> List[ParsedChunk]:
        """
        Parse a single file and return chunks.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            List of ParsedChunk objects
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {path.suffix}")
            
            logger.info(f"Parsing file: {file_path}")
            
            # Detect file type
            file_type = self._detect_file_type(path)
            
            # Handle PDF separately (binary file)
            if file_type == 'pdf':
                text = self._parse_pdf(path)
            else:
                # Read text file content
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse based on file type
                if file_type == 'markdown':
                    text = self._parse_markdown(content)
                elif file_type == 'rst':
                    text = self._parse_rst(content)
                elif file_type == 'html':
                    text = self._parse_html(content)
                else:  # plain text
                    text = content
            
            # Normalize
            text = self._normalize_text(text)
            
            if not text:
                logger.warning(f"No content extracted from {file_path}")
                return []
            
            # Apply chunking strategy
            if self.chunking_strategy == 'fixed':
                chunks = self._chunk_fixed(text, str(path))
            elif self.chunking_strategy == 'semantic':
                chunks = self._chunk_semantic(text, str(path))
            elif self.chunking_strategy == 'sliding':
                chunks = self._chunk_sliding(text, str(path))
            else:
                raise ValueError(f"Unknown chunking strategy: {self.chunking_strategy}")
            
            # Add file metadata to all chunks
            for chunk in chunks:
                chunk.metadata.update({
                    'file_type': file_type,
                    'file_name': path.name,
                    'file_size': path.stat().st_size
                })
            
            logger.info(f"Parsed {path.name}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            raise CustomException(f"Failed to parse {file_path}: {e}", sys)
    
    def parse_directory(
        self,
        directory: str,
        recursive: bool = True,
        extensions: Optional[List[str]] = None
    ) -> List[ParsedChunk]:
        """
        Parse all supported files in a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories
            extensions: Filter by specific extensions (e.g., ['.md', '.rst'])
            
        Returns:
            List of all ParsedChunk objects from all files
        """
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")
            
            if not dir_path.is_dir():
                raise ValueError(f"Not a directory: {directory}")
            
            # Determine which extensions to look for
            target_extensions = set(extensions) if extensions else self.SUPPORTED_EXTENSIONS
            
            # Find all matching files
            all_chunks = []
            pattern = '**/*' if recursive else '*'
            
            for file_path in dir_path.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in target_extensions:
                    try:
                        chunks = self.parse_file(str(file_path))
                        all_chunks.extend(chunks)
                    except Exception as e:
                        logger.error(f"Skipping {file_path}: {e}")
                        continue
            
            logger.info(f"Parsed directory {directory}: {len(all_chunks)} total chunks")
            return all_chunks
            
        except Exception as e:
            raise CustomException(f"Failed to parse directory {directory}: {e}", sys)
    
    def get_stats(self, chunks: List[ParsedChunk]) -> Dict[str, Any]:
        """Get statistics about parsed chunks."""
        if not chunks:
            return {"total_chunks": 0}
        
        chunk_types = {}
        strategies = {}
        total_chars = 0
        
        for chunk in chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
            strategy = chunk.metadata.get('strategy', 'unknown')
            strategies[strategy] = strategies.get(strategy, 0) + 1
            total_chars += len(chunk.content)
        
        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "avg_chunk_size": total_chars // len(chunks),
            "chunk_types": chunk_types,
            "strategies_used": strategies,
            "unique_files": len(set(c.source_file for c in chunks))
        }


# Convenience function for quick parsing
def parse_documents(
    path: str,
    chunking_strategy: str = 'semantic',
    chunk_size: int = 512
) -> List[ParsedChunk]:
    """
    Quick helper to parse documents from a file or directory.
    
    Args:
        path: File or directory path
        chunking_strategy: 'fixed', 'semantic', or 'sliding'
        chunk_size: Target chunk size
        
    Returns:
        List of ParsedChunk objects
    """
    parser = DocumentParser(
        chunk_size=chunk_size,
        chunking_strategy=chunking_strategy
    )
    
    path_obj = Path(path)
    
    if path_obj.is_file():
        return parser.parse_file(path)
    elif path_obj.is_dir():
        return parser.parse_directory(path)
    else:
        raise ValueError(f"Path does not exist: {path}")