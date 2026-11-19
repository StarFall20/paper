"""Screen named sources locally for contiguous prose overlap."""

import argparse
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re
import unicodedata

from pypdf import PdfReader


def tokens(value):
    text = unicodedata.normalize("NFKC", value).casefold().replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "", text)
    return tuple(re.findall(r"[^\W\d_]+", text))


def draft_blocks(path):
    value = path.read_text().split("## References", 1)[0]
    value = re.sub(r"\$\$[\s\S]*?\$\$|\$[^\n$]+\$", "\n\n", value)
    for line, block in enumerate(re.split(r"\n\s*\n", value), 1):
        if block.strip() and not block.lstrip().startswith(("#", "|", "!", "`")):
            yield line, tokens(block)


def compare(pages, paragraphs, minimum):
    index = defaultdict(list)
    for n, page in enumerate(pages, 1):
        for i in range(len(page) - minimum + 1):
            index[page[i:i + minimum]].append((n, i))
    found = {}
    normalized_phrases = {}
    for source_paragraph, text in paragraphs:
        for j in range(len(text) - minimum + 1):
            for page_no, i in index.get(text[j:j + minimum], ()):
                page = pages[page_no - 1]
                if j and i and text[j - 1] == page[i - 1]:
                    continue
                size = minimum
                while j + size < len(text) and i + size < len(page) and text[j + size] == page[i + size]:
                    size += 1
                found[source_paragraph, j, page_no, i] = size
                normalized_phrases[source_paragraph, j, page_no, i] = " ".join(text[j:j + size])
    return [{ "paragraph": paragraph, "draft_token_offset": j, "source_page": p,
              "source_token_offset": i, "matching_words": count}
            for (paragraph, j, p, i), count in sorted(found.items())], normalized_phrases


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drafts", nargs="+", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--ppi-preprint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--private-excerpts", type=Path)
    args = p.parse_args()
    metadata = json.loads((args.source_dir / "acquisition.json").read_text())
    paths = {key: args.source_dir / (key + ".pdf") for key, value in metadata.items()
             if value["readable_pdf"]}
    paths["ppi_plus_plus_v2"] = args.ppi_preprint
    metadata["ppi_plus_plus_v2"] = {"url": "https://arxiv.org/pdf/2311.01453v2"}
    drafts = {p.name: list(draft_blocks(p)) for p in args.drafts}
    sources, phrases = {}, {}
    for key, path in paths.items():
        sha = digest(path)
        assert metadata[key].get("sha256", sha) == sha
        reader = PdfReader(path, strict=True)
        pages = [tokens(page.extract_text() or "") for page in reader.pages]
        assert sum(map(len, pages)) >= 1000
        sources[key] = {"url": metadata[key]["url"], "sha256": sha, "pages": len(pages),
                        "nonempty_pages": sum(bool(page) for page in pages), "comparisons": {}}
        phrases[key] = {}
        for draft_name, paragraphs in drafts.items():
            hits, audit = compare(pages, paragraphs, 10)
            sources[key]["comparisons"][draft_name] = hits
            phrases[key][draft_name] = list(audit.values())
    result = {"state": "passed", "minimum_contiguous_words": 10,
              "extractor": {"pypdf": version("pypdf"), "fonttools": version("fonttools")},
              "scope": "Reported exact token matches in extracted text from the named source versions. Candidate references, tables, images and equations are excluded. Semantic similarity and other sources require other evidence.",
              "draft_sha256": {path.name: digest(path) for path in args.drafts},
              "sources": sources,
              "unavailable_primary_downloads": {key: value for key, value in metadata.items()
                                                if key not in paths and not value.get("valid_zip", False)}}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.private_excerpts:
        args.private_excerpts.write_text(json.dumps(phrases, indent=2) + "\n")
    print(json.dumps({"screened_sources": len(sources),
                      "source_page_count": sum(s["pages"] for s in sources.values()),
                      "hits": {key: {name: len(hits) for name, hits in s["comparisons"].items()}
                               for key, s in sources.items()}}))


if __name__ == "__main__":
    main()
