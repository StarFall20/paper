"""Screen manuscript prose against locally exported Web of Science abstracts."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata

START = re.compile(r"^([A-Z0-9]{2}) (.*)$")
WORDS = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
MIN_WORDS = 8


def fields(bundle):
    result = {}
    key = None
    for line in bundle.splitlines():
        field = START.match(line)
        if field:
            key = field[1]
            result.setdefault(key, []).append(field[2])
        elif key is not None and line.startswith("   "):
            result[key].append(line.strip())
    return {key: " ".join(items) for key, items in result.items()}


def words(text):
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return tuple(WORDS.findall(normalized))


def prose(path):
    text = path.read_text().split("\n## References\n", 1)[0]
    text = re.sub(r"\$\$.*?\$\$", "\n", text, flags=re.S)
    for index, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith(("#", "|", "!", "```")):
            continue
        for chunk in re.split(r"\$[^$\n]+\$", line):
            items = words(chunk)
            if len(items) >= MIN_WORDS:
                yield index, items


def scan(sources, drafts, out, inspection):
    index = defaultdict(list)
    source_text = {}
    metadata = {}
    fingerprints = []
    all_records = set()
    for source in sources:
        blob = source.read_bytes()
        records = [fields(entry)
                   for entry in re.split(r"^ER\s*$", blob.decode("utf-8-sig"), flags=re.M)]
        records = [row for row in records if row.get("UT", "").strip()]
        dig = sha256(blob).hexdigest()
        fingerprints.append(dict(name=source.name, sha256=dig, records=len(records),
                                 with_abstract=sum(bool(row.get("AB", "").strip()) for row in records)))
        for row in records:
            uid = row["UT"]
            all_records.add(uid)
            summary = words(row.get("AB", ""))
            if len(summary) < MIN_WORDS:
                continue
            key = (uid, summary)
            if key in source_text:
                continue
            source_text[key] = summary
            metadata[key] = (row.get("DI", ""), row.get("TI", ""))
    for key, reference in source_text.items():
        for pos in range(len(reference) - MIN_WORDS + 1):
            index[reference[pos:pos + MIN_WORDS]].append((key, pos))
    overall = []
    for draft in drafts:
        best = {}
        for line, candidate in prose(draft):
            for i in range(len(candidate) - MIN_WORDS + 1):
                for key, j in index.get(candidate[i:i+MIN_WORDS], ()):
                    source = source_text[key]
                    width = MIN_WORDS
                    while i + width < len(candidate) and j + width < len(source) and candidate[i + width] == source[j + width]:
                        width += 1
                    uid = key[0]
                    old = best.get(uid)
                    if old is None or width > old["words"]:
                        doi, title = metadata[key]
                        best[uid] = dict(wos_id=uid, doi=doi, title=title, words=width, manuscript_line=line,
                                         phrase=candidate[i:i+width])
        matches = sorted(best.values(), key=lambda item: (-item["words"], item["wos_id"]))
        overall.append(dict(draft=draft.name, sha256=sha256(draft.read_bytes()).hexdigest(),
                            match_length_distribution=dict(Counter(x["words"] for x in matches)),
                            matches=matches))
    unique_wos = {key[0] for key in source_text}
    payload = dict(
        source_exports=fingerprints, unique_records=len(all_records), unique_records_with_abstract=len(unique_wos),
        distinct_source_abstracts=len(source_text), min_match_words=MIN_WORDS,
        scope="Locally archived abstract texts in 12 Web of Science exports, matched to prose. "
              "Excludes reference lists, equations, headings and tables. The screen records contiguous "
              "matches of normalized tokens of length eight or more. No third-party manuscript upload.",
        checked_drafts=overall,
    )
    if inspection is not None:
        inspection.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    for check in overall:
        for record in check["matches"]:
            record.pop("phrase")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(records=len(all_records), abstracts=len(unique_wos),
                          abstract_versions=len(source_text),
                          match_counts={a["draft"]: a["match_length_distribution"] for a in overall})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inspection", type=Path)
    args = parser.parse_args()
    files = [file for folder in args.input_dir for file in sorted(folder.glob("*.txt"))]
    assert len(files) == 12
    drafts = [args.draft_dir / name for name in ("manuscript.md", "supplement.md")]
    scan(files, drafts, args.out, args.inspection)
