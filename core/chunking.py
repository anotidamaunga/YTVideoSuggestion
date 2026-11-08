import re
from typing import List

heading_re = re.compile(r"^#{1,6}\s+\S")

def split_blocks(text: str) -> List[str]:
    blocks = []
    for raw_block in re.split(r"\n\s*\n", text):
        block = raw_block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if len(lines) > 1 and heading_re.match(lines[0]):
            blocks.append(lines[0])
            remainder = "\n".join(lines[1:]).strip()
            if remainder:
                blocks.append(remainder)
        else:
            blocks.append(block)
    return blocks

def chunk_text(text: str, max_words: int = 600, overlap_words: int = 80) -> list[str]:
    blocks = split_blocks(text)
    if not blocks:
        return []

    chunks: list[str] = []
    current_words: list[str] = []

    for block in blocks:
        block_words = block.split()
        if len(current_words) + len(block_words) > max_words and current_words:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap_words:] if overlap_words else []
        current_words.extend(block_words)

        # A single block larger than max_words: flush it as its own chunk.
        while len(current_words) > max_words:
            chunks.append(" ".join(current_words[:max_words]))
            current_words = current_words[max_words - overlap_words:] if overlap_words else current_words[max_words:]

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks
