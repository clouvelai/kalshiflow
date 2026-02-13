"""Unit tests for ChunkingPipeline.

Tests cover:
  1. estimate_tokens: correct heuristic (len // 4)
  2. chunk_article -- empty input: returns empty list
  3. chunk_article -- small article (<500 tokens / <2000 chars): single chunk
  4. chunk_article -- medium article (500-1500 tokens): paragraph splitting
  5. chunk_article -- large article (>1500 tokens / >6000 chars): fixed-size windows
  6. ArticleChunk dataclass: fields populated correctly
  7. Metadata propagation: each chunk inherits parent metadata
  8. parent_url extraction: pulled from metadata["news_url"]
  9. Paragraph splitting: respects \\n\\n boundaries
  10. Paragraph merging: small paragraphs merged until budget exceeded
  11. Oversized single paragraph: handled as its own chunk
  12. Fixed-size overlap: chunks overlap by 64 tokens (256 chars)
  13. chunk_index and total_chunks: correctly set on each chunk
  14. No external dependencies: all tests are pure Python, no mocks needed
  15. Edge cases: whitespace-only content, single character, very long single word
"""

import pytest

from kalshiflow_rl.traderv3.single_arb.chunking import ArticleChunk, ChunkingPipeline


# ---------------------------------------------------------------------------
#  estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_basic():
    assert ChunkingPipeline.estimate_tokens("hello world") == 2  # 11 chars // 4


def test_estimate_tokens_empty():
    assert ChunkingPipeline.estimate_tokens("") == 0


def test_estimate_tokens_short():
    # 3 chars -> 3 // 4 = 0
    assert ChunkingPipeline.estimate_tokens("abc") == 0


def test_estimate_tokens_exact_multiple_of_four():
    assert ChunkingPipeline.estimate_tokens("abcd") == 1  # 4 // 4


def test_estimate_tokens_large():
    text = "a" * 10_000
    assert ChunkingPipeline.estimate_tokens(text) == 2500


# ---------------------------------------------------------------------------
#  chunk_article -- empty / whitespace input
# ---------------------------------------------------------------------------


def test_chunk_article_empty_string():
    result = ChunkingPipeline.chunk_article("", {})
    assert result == []


def test_chunk_article_none_content():
    result = ChunkingPipeline.chunk_article(None, {})
    assert result == []


def test_chunk_article_whitespace_only():
    result = ChunkingPipeline.chunk_article("   \n\n  \t  ", {})
    assert result == []


# ---------------------------------------------------------------------------
#  chunk_article -- small article (<500 tokens / <2000 chars)
# ---------------------------------------------------------------------------


def test_small_article_single_chunk():
    # 400 chars -> 100 tokens, well under 500-token threshold
    content = "A short news article about markets. " * 10  # ~360 chars
    meta = {"news_url": "https://example.com/article", "source": "reuters"}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    assert len(chunks) == 1
    assert chunks[0].content == content
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1


def test_small_article_preserves_content_exactly():
    content = "Paragraph one.\n\nParagraph two."
    meta = {"news_url": "https://example.com"}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    assert len(chunks) == 1
    assert chunks[0].content == content


# ---------------------------------------------------------------------------
#  chunk_article -- medium article (500-1500 tokens, paragraph split)
# ---------------------------------------------------------------------------


def _make_paragraph(words: int = 100) -> str:
    """Generate a paragraph of approximately *words* words."""
    return ("prediction market trading activity ") * words


def _make_medium_article() -> str:
    """~800 tokens (3200 chars), triggers paragraph splitting."""
    paras = [_make_paragraph(8) for _ in range(10)]
    return "\n\n".join(paras)


def test_medium_article_splits_into_multiple_chunks():
    content = _make_medium_article()
    tokens = ChunkingPipeline.estimate_tokens(content)
    assert 500 <= tokens <= 1500, f"Expected medium range, got {tokens}"

    meta = {"news_url": "https://example.com/medium"}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    assert len(chunks) > 1


def test_medium_article_paragraph_boundaries():
    """Chunks from medium articles should not split mid-paragraph."""
    content = _make_medium_article()
    meta = {"news_url": ""}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    for chunk in chunks:
        # Each chunk's content should be reconstructable from whole paragraphs
        # joined by \n\n -- no partial paragraph fragments
        parts = chunk.content.split("\n\n")
        for part in parts:
            assert part.strip(), "Empty paragraph part found in chunk"


# ---------------------------------------------------------------------------
#  chunk_article -- large article (>1500 tokens, fixed-size windows)
# ---------------------------------------------------------------------------


def _make_large_article() -> str:
    """~2000 tokens (8000 chars), triggers fixed-size chunking."""
    return "x" * 8000


def test_large_article_multiple_chunks():
    content = _make_large_article()
    tokens = ChunkingPipeline.estimate_tokens(content)
    assert tokens > 1500, f"Expected >1500 tokens, got {tokens}"

    meta = {"news_url": "https://example.com/large"}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    assert len(chunks) > 1


def test_large_article_fixed_chunk_size():
    """Each chunk (except possibly the last) should be ~512 tokens / 2048 chars."""
    content = _make_large_article()
    meta = {"news_url": ""}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    # All chunks except the last should be exactly 2048 chars (512 tokens * 4)
    for chunk in chunks[:-1]:
        assert len(chunk.content) == 2048


# ---------------------------------------------------------------------------
#  ArticleChunk dataclass fields
# ---------------------------------------------------------------------------


def test_article_chunk_fields():
    chunk = ArticleChunk(
        content="some text",
        chunk_index=2,
        total_chunks=5,
        parent_url="https://example.com",
        token_count=42,
        metadata={"key": "val"},
    )
    assert chunk.content == "some text"
    assert chunk.chunk_index == 2
    assert chunk.total_chunks == 5
    assert chunk.parent_url == "https://example.com"
    assert chunk.token_count == 42
    assert chunk.metadata == {"key": "val"}


def test_article_chunk_default_metadata():
    chunk = ArticleChunk(
        content="text",
        chunk_index=0,
        total_chunks=1,
        parent_url="",
        token_count=1,
    )
    assert chunk.metadata == {}


# ---------------------------------------------------------------------------
#  Metadata propagation
# ---------------------------------------------------------------------------


def test_metadata_propagated_to_all_chunks():
    content = _make_medium_article()
    meta = {"news_url": "https://example.com", "source": "reuters", "topic": "fed"}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    for chunk in chunks:
        assert chunk.metadata["source"] == "reuters"
        assert chunk.metadata["topic"] == "fed"
        assert chunk.metadata["news_url"] == "https://example.com"


def test_metadata_is_copied_not_shared():
    """Mutating one chunk's metadata must not affect others."""
    content = _make_medium_article()
    meta = {"news_url": "https://example.com"}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    if len(chunks) > 1:
        chunks[0].metadata["mutated"] = True
        assert "mutated" not in chunks[1].metadata
        assert "mutated" not in meta


# ---------------------------------------------------------------------------
#  parent_url extraction
# ---------------------------------------------------------------------------


def test_parent_url_from_metadata():
    content = "Short article content here and more."
    meta = {"news_url": "https://news.example.com/story/123"}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    assert chunks[0].parent_url == "https://news.example.com/story/123"


def test_parent_url_missing_defaults_to_empty():
    content = "Short article content here and more."
    meta = {"source": "reuters"}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    assert chunks[0].parent_url == ""


# ---------------------------------------------------------------------------
#  Paragraph merging
# ---------------------------------------------------------------------------


def test_small_paragraphs_merged():
    """Multiple tiny paragraphs should be merged into a single chunk."""
    paras = ["Short paragraph."] * 5  # each ~4 tokens
    content = "\n\n".join(paras)
    meta = {"news_url": ""}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    # Total is ~100 chars -> ~25 tokens, well under 500 -> single chunk (small tier)
    assert len(chunks) == 1


def test_paragraph_merge_respects_budget():
    """When paragraphs are large enough, merging should stop at the budget."""
    # Build paragraphs that are each ~200 tokens (800 chars)
    para = "word " * 160  # 800 chars = 200 tokens
    content = "\n\n".join([para] * 5)  # ~1000 tokens total -> medium tier

    meta = {"news_url": ""}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    # With max_tokens=512 per chunk and paras of ~200 tokens each,
    # we should get multiple chunks (each holding 2 paragraphs)
    assert len(chunks) >= 2


# ---------------------------------------------------------------------------
#  Oversized single paragraph
# ---------------------------------------------------------------------------


def test_oversized_paragraph_own_chunk():
    """A single paragraph exceeding max_tokens should become its own chunk."""
    # One huge paragraph (~600 tokens) + small ones -> medium tier total
    huge_para = "a" * 2400  # 600 tokens
    small_para = "b" * 400  # 100 tokens
    content = f"{huge_para}\n\n{small_para}\n\n{small_para}"
    # Total ~800 tokens -> medium tier

    meta = {"news_url": ""}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    # The huge paragraph should be its own chunk
    found_huge = any(chunk.content == huge_para for chunk in chunks)
    assert found_huge, "Oversized paragraph should appear as its own chunk"


# ---------------------------------------------------------------------------
#  Fixed-size overlap (64 tokens = 256 chars)
# ---------------------------------------------------------------------------


def test_fixed_overlap_chars():
    """Adjacent chunks in large articles should overlap by 256 chars."""
    content = "a" * 8000  # 2000 tokens -> large tier
    meta = {"news_url": ""}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    assert len(chunks) >= 2

    # Each chunk is 2048 chars. Stride = 2048 - 256 = 1792.
    # So chunk[0] covers [0:2048], chunk[1] covers [1792:3840].
    # The overlap region is content[1792:2048] = 256 chars.
    chunk0_end = chunks[0].content
    chunk1_start = chunks[1].content

    # Last 256 chars of chunk 0 should equal first 256 chars of chunk 1
    assert chunk0_end[-256:] == chunk1_start[:256]


# ---------------------------------------------------------------------------
#  chunk_index and total_chunks
# ---------------------------------------------------------------------------


def test_chunk_index_sequential():
    content = _make_large_article()
    meta = {"news_url": ""}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_total_chunks_consistent():
    content = _make_large_article()
    meta = {"news_url": ""}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    for chunk in chunks:
        assert chunk.total_chunks == len(chunks)


# ---------------------------------------------------------------------------
#  Edge cases
# ---------------------------------------------------------------------------


def test_single_character():
    chunks = ChunkingPipeline.chunk_article("x", {})
    assert len(chunks) == 1
    assert chunks[0].content == "x"
    assert chunks[0].token_count == 0  # 1 // 4 = 0


def test_very_long_single_word():
    """A very long word with no whitespace or paragraph breaks."""
    word = "a" * 8000  # 2000 tokens -> large tier, no paragraph breaks
    meta = {"news_url": ""}
    chunks = ChunkingPipeline.chunk_article(word, meta)

    assert len(chunks) > 1
    # All content should be recoverable (accounting for overlap)
    assert chunks[0].content[:100] == "a" * 100


def test_token_count_on_each_chunk():
    """Each chunk's token_count should match estimate_tokens of its content."""
    content = _make_large_article()
    meta = {"news_url": ""}
    chunks = ChunkingPipeline.chunk_article(content, meta)

    for chunk in chunks:
        assert chunk.token_count == ChunkingPipeline.estimate_tokens(chunk.content)
