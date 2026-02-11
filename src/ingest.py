import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pypdf import PdfReader
from rich.console import Console
from striprtf.striprtf import rtf_to_text
import argparse

from .config import paths, settings

console = Console()


@dataclass
class Chunk:
    text: str
    section: Optional[str]
    heading: Optional[str]
    part: Optional[str]
    schedule: Optional[str]
    division: Optional[str]
    subdivision: Optional[str]
    top_level_block: Optional[str]
    top_level_kind: Optional[str]
    top_level_id: Optional[str]
    source: str

    def to_json(self, idx: int) -> str:
        payload = {
            "id": idx,
            "text": self.text.strip(),
            "section": self.section,
            "heading": self.heading,
            "part": self.part,
            "schedule": self.schedule,
            "division": self.division,
            "subdivision": self.subdivision,
            "top_level_block": self.top_level_block,
            "top_level_kind": self.top_level_kind,
            "top_level_id": self.top_level_id,
            "source": self.source,
            "tokens": len(self.text.split()),
        }
        return json.dumps(payload, ensure_ascii=False)


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n".join(pages)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_rtf(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return rtf_to_text(raw)


def clean(text: str) -> str:
    # Normalize spacing while preserving paragraph breaks; fix common PDF line artifacts.
    text = text.replace("\r", "")
    # Join hyphenated breaks (e.g., “cust-\nomer” -> “customer”).
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # Join linebreak-split words without hyphens (e.g., "cus\nstomer" -> "customer"),
    # but keep line breaks before likely legal headings so section parsing remains stable.
    text = re.sub(
        r"(?<=\w)\n(?=(?!\d+[A-Z]{0,4}\.?\s)(?!Schedule\s+\d)(?!Part\s+\d)(?!Division\s+\d)(?!Subdivision\s+\d)\w)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    # Collapse extra spaces and blank lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


SECTION_PATTERN = re.compile(
    r"^\s*(?P<num>(?:\d{1,3}[A-Z]{0,3}|53Z[A-Z]{1,3}))\.\s*(?P<title>[A-Z][^\n]{1,300})",
    re.MULTILINE,
)
SCHEDULE_PATTERN = re.compile(
    r"^\s*(?P<header>Schedule\s+(?P<num>\d+[A-Za-z]*))\s*(?P<title>[^\n]{1,300})",
    re.MULTILINE,
)
DIVISION_PATTERN = re.compile(r"^\s*(Division\s+\d+[A-Za-z0-9]*\s*[—\-–]\s*[^\n]+)", re.MULTILINE)
SUBDIVISION_PATTERN = re.compile(r"^\s*(Subdivision\s+\d+[A-Za-z0-9]*\s*[—\-–]\s*[^\n]+)", re.MULTILINE)
PART_HEADER_PATTERN = re.compile(r"^\s*(Part\s+\d+[A-Za-z]*)\b[^\n]*", re.MULTILINE)

# Top-level Part/Schedule headings in Cap. 615, used to infer
# which Schedule a numbered paragraph belongs to.
TOP_LEVEL_BLOCK_PATTERN = re.compile(
    r"^\s*(?P<kind>Part|Schedule)\s+(?P<id>\d+[A-Za-z]*)\b[^\n]*",
    re.MULTILINE,
)
MAIN_BODY_START_PATTERN = re.compile(r"\bPart\s+1\s+Preliminary\b", re.IGNORECASE)
ORDINANCE_BODY_ANCHOR_PATTERN = re.compile(r"\bAn Ordinance to provide\b", re.IGNORECASE)

ALLOWED_SCHEDULE_BLOCKS = {
    ("Schedule", "2"),
    ("Schedule", "3B"),
}
PART5B_BLOCK = ("Part", "5B")

# Within Part 5B, we further restrict to these sections:
ALLOWED_PART5B_SECTIONS = {
    "53ZRD",
    "53ZRE",
    "53ZRF",
    "53ZRG",
    "53ZRH",
    "53ZRI",
    "53ZRJ",
    "53ZRK",
    "53ZRL",
}


def _split_sections_internal(
    text: str,
) -> List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], int]]:
    """
    Split text into sections using Cap. 615-style numbering (e.g., 53ZRD.).
    Track Division/Subdivision headers for context.

    Returns tuples of (section_id, heading, body, part, division, subdivision, header_pos).
    """
    matches = list(SECTION_PATTERN.finditer(text)) + list(SCHEDULE_PATTERN.finditer(text))
    matches = sorted(matches, key=lambda m: m.start())
    if not matches:
        return [(None, None, text, None, None, None, 0)]

    parts = sorted([(m.start(), m.group(1).strip()) for m in PART_HEADER_PATTERN.finditer(text)], key=lambda x: x[0])
    divisions = sorted([(m.start(), m.group(1).strip()) for m in DIVISION_PATTERN.finditer(text)], key=lambda x: x[0])
    subdivisions = sorted(
        [(m.start(), m.group(1).strip()) for m in SUBDIVISION_PATTERN.finditer(text)],
        key=lambda x: x[0],
    )

    def latest_heading(pos: int, items: List[Tuple[int, str]]) -> Optional[str]:
        active = [h for start, h in items if start <= pos]
        return active[-1] if active else None

    slices: List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], int]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        header = match.groupdict().get("header") or match.groupdict().get("num") or ""
        heading_raw = match.groupdict().get("title") or ""
        section_id = header.strip().rstrip(".")
        heading = heading_raw.strip(" -–—.")

        header_pos = match.start()
        part = latest_heading(header_pos, parts)
        div = latest_heading(header_pos, divisions)
        subdiv = latest_heading(header_pos, subdivisions)

        prefix_lines = []
        if part:
            prefix_lines.append(part)
        if div:
            prefix_lines.append(div)
        if subdiv:
            prefix_lines.append(subdiv)
        header_line = " ".join([p for p in [header, heading] if p])
        if header_line:
            prefix_lines.append(header_line)
        content = "\n".join(prefix_lines + [body]).strip()
        slices.append((section_id, heading, content, part, div, subdiv, header_pos))
    return slices


def split_sections(text: str) -> List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]]:
    """
    Public helper used elsewhere: like _split_sections_internal, but without
    exposing header positions.
    """
    raw = _split_sections_internal(text)
    return [(sid, heading, body, div, subdiv) for sid, heading, body, _, div, subdiv, _ in raw]


def split_sections_with_positions(
    text: str,
) -> List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], int]]:
    """Variant that also returns the header position for each section."""
    return _split_sections_internal(text)


def chunk_text(
    section_id: Optional[str],
    heading: Optional[str],
    text: str,
    part: Optional[str],
    schedule: Optional[str],
    division: Optional[str],
    subdivision: Optional[str],
    top_level_block: Optional[str],
    top_level_kind: Optional[str],
    top_level_id: Optional[str],
) -> Iterable[Chunk]:
    """Section-aware chunking with soft token sizing."""
    tokens = text.split()
    max_tokens = settings.max_chunk_tokens
    min_tokens = settings.min_chunk_tokens
    overlap = settings.overlap_tokens

    if len(tokens) <= max_tokens:
        yield Chunk(
            text=text,
            section=section_id,
            heading=heading,
            part=part,
            schedule=schedule,
            division=division,
            subdivision=subdivision,
            top_level_block=top_level_block,
            top_level_kind=top_level_kind,
            top_level_id=top_level_id,
            source="amlo_cap615",
        )
        return

    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        slice_tokens = tokens[start:end]
        if len(slice_tokens) < min_tokens and start != 0:
            break
        chunk_text_val = " ".join(slice_tokens)
        yield Chunk(
            text=chunk_text_val,
            section=section_id,
            heading=heading,
            part=part,
            schedule=schedule,
            division=division,
            subdivision=subdivision,
            top_level_block=top_level_block,
            top_level_kind=top_level_kind,
            top_level_id=top_level_id,
            source="amlo_cap615",
        )
        if end == len(tokens):
            break
        start = end - overlap


def _block_label(block: Optional[Tuple[str, str]]) -> Optional[str]:
    if not block:
        return None
    return f"{block[0]} {block[1]}"


def _is_toc_like(text: str) -> bool:
    t = text.lower()
    return ("section page" in t) or (" table of contents" in t)


def trim_front_matter(text: str) -> str:
    """
    Drop pre-body table-of-contents/front matter when possible.
    This improves section/block inference by anchoring parsing at
    the first actual ordinance part heading.
    """
    m = ORDINANCE_BODY_ANCHOR_PATTERN.search(text)
    if m:
        return text[m.start() :]
    m = MAIN_BODY_START_PATTERN.search(text)
    if not m:
        return text
    return text[m.start() :]


def ingest(source: Path, out_path: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Cannot find source file: {source}")

    suffix = source.suffix.lower()
    if suffix == ".pdf":
        raw_text = read_pdf(source)
    elif suffix == ".rtf":
        raw_text = read_rtf(source)
    else:
        raw_text = read_text(source)

    cleaned = clean(raw_text)
    working_text = trim_front_matter(cleaned)
    sections = split_sections_with_positions(working_text)

    # Build explicit spans for Schedule 2 and 3B using all schedule headings
    # as boundaries, so we do not accidentally classify later schedules.
    all_schedule_headers: List[Tuple[int, str]] = []
    for m in SCHEDULE_PATTERN.finditer(working_text):
        sched_num = (m.groupdict().get("num") or "").strip().upper()
        if not sched_num:
            continue
        all_schedule_headers.append((m.start(), sched_num))
    all_schedule_headers.sort(key=lambda x: x[0])

    target_schedule_spans: List[Tuple[int, int, str]] = []
    for idx, (start, sched_num) in enumerate(all_schedule_headers):
        end = all_schedule_headers[idx + 1][0] if idx + 1 < len(all_schedule_headers) else len(working_text)
        if sched_num in {"2", "3B"}:
            target_schedule_spans.append((start, end, f"Schedule {sched_num}"))

    def schedule_for(pos: int) -> Optional[str]:
        for start, end, label in target_schedule_spans:
            if start <= pos < end:
                return label
        return None

    schedule_rows: List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], int]] = []
    part5b_candidates: Dict[str, List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], int]]] = {
        sec: [] for sec in ALLOWED_PART5B_SECTIONS
    }

    for section_id, heading, body, part, division, subdivision, header_pos in sections:
        # Skip unnumbered blobs.
        if section_id is None:
            continue

        # Capture Part 5B section candidates; we will select the best body
        # for each section ID to avoid table-of-contents fragments.
        if section_id in ALLOWED_PART5B_SECTIONS:
            part5b_candidates[section_id].append(
                (section_id, heading, body, part, division, subdivision, "Part 5B", header_pos)
            )
            continue

        sched_label = schedule_for(header_pos)
        # Keep anything inside Schedule 2 or Schedule 3B (headers and
        # their numbered paragraphs), while filtering obvious parser noise.
        if sched_label:
            body_tokens = len((body or "").split())
            # Repeated page header lines often appear as tiny "Schedule X"
            # rows; keep substantive content only.
            if str(section_id).startswith("Schedule") and body_tokens < 25:
                continue
            # Inside schedules, section IDs like 53ZRA are usually leakage
            # from running headers/cross-references, not schedule paragraphs.
            if re.match(r"^53Z[A-Z]+$", str(section_id)):
                continue
            schedule_rows.append((section_id, heading, body, part, division, subdivision, sched_label, header_pos))

    selected_rows: List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], int]] = []
    selected_rows.extend(schedule_rows)

    # For each target Part 5B section, keep the best candidate (longest
    # non-TOC-like body), prioritizing candidates that appear inside Part 5B.
    missing_part5b = []
    for sec_id in sorted(ALLOWED_PART5B_SECTIONS):
        candidates = part5b_candidates.get(sec_id, [])
        if not candidates:
            missing_part5b.append(sec_id)
            continue

        def quality(
            row: Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], int]
        ) -> Tuple[int, int, int]:
            _, _, body, _, _, _, label, _ = row
            body_text = body or ""
            tok = len(body_text.split())
            in_part5b = 1 if (label == _block_label(PART5B_BLOCK) or part == "Part 5B") else 0
            toc_penalty = 1 if _is_toc_like(body_text) else 0
            # Prefer in-Part5B, longer text, and non-TOC bodies.
            return (in_part5b, tok, -toc_penalty)

        best = max(candidates, key=quality)
        selected_rows.append(best)

    if missing_part5b:
        console.print(
            f"[yellow]Warning: missing target Part 5B sections in parsed text: {', '.join(missing_part5b)}"
        )

    selected_rows.sort(key=lambda row: row[-1])

    chunks: List[Chunk] = []
    for section_id, heading, body, part, division, subdivision, top_level_block, _ in selected_rows:
        top_level_kind = None
        top_level_id = None
        if top_level_block:
            m = re.match(r"^(Part|Schedule)\s+(\d+[A-Za-z]*)$", top_level_block)
            if m:
                top_level_kind = m.group(1)
                top_level_id = m.group(2).upper()
        part_value = (part or top_level_block) if top_level_kind == "Part" else None
        schedule_value = top_level_block if top_level_kind == "Schedule" else None
        # Division/Subdivision headings are meaningful for Part-based content.
        # For schedules, they are often absent or noisy in PDF extraction.
        division_value = division if top_level_kind == "Part" else None
        subdivision_value = subdivision if top_level_kind == "Part" else None
        for ch in chunk_text(
            section_id,
            heading,
            body,
            part_value,
            schedule_value,
            division_value,
            subdivision_value,
            top_level_block,
            top_level_kind,
            top_level_id,
        ):
            chunks.append(ch)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            f.write(chunk.to_json(idx) + "\n")

    console.print(f"[bold green]Wrote {len(chunks)} chunks to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Ingest and chunk Cap. 615 (pdf/rtf/txt).")
    parser.add_argument("--source", required=True, type=Path, help="Path to source document")
    parser.add_argument("--out", default=paths.default_chunks, type=Path, help="Output JSONL path")
    parser.add_argument("--min_tokens", type=int, default=settings.min_chunk_tokens, help="Minimum tokens per chunk")
    parser.add_argument("--max_tokens", type=int, default=settings.max_chunk_tokens, help="Maximum tokens per chunk")
    parser.add_argument("--overlap", type=int, default=settings.overlap_tokens, help="Overlap tokens between chunks")
    args = parser.parse_args()

    settings.min_chunk_tokens = args.min_tokens
    settings.max_chunk_tokens = args.max_tokens
    settings.overlap_tokens = args.overlap
    ingest(args.source, args.out)


if __name__ == "__main__":
    main()
