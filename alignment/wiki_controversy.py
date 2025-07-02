import re
import argparse
from datasets import Dataset
from pathlib import Path
from datasets import load_dataset
import logging

# --------------------------------------------------------------------------- #
# 1.  ≤ 50 regexes that flag potentially contentious section titles
_PATTERNS = [
    r'\bCriticism?s?\b', r'\bControvers(?:y|ies)\b', r'\bDisputes?\b',
    r'\bDebates?\b', r'\bAllegations?\b', r'\bScandals?\b', r'\bBacklash\b',
    r'\bOpposition\b', r'\bProtests?\b', r'\bBoycott\b', r'\bComplaints?\b',
    r'\bDissent\b', r'\bLegal (issues|challenges)\b', r'\bLawsuits?\b',
    r'\bLitigation\b', r'\bMisconduct\b', r'\bCorruption\b', r'\bFraud\b',
    r'\bBribery\b', r'\bPrice[ -]?fixing\b', r'\bAbuse\b', r'\bHarassment\b',
    r'\bSexual (misconduct|assault)\b', r'\bRacism\b', r'\bSexism\b',
    r'\bXenophobia\b', r'\bDiscrimination\b', r'\bPlagiarism\b',
    r'\bCensorship\b', r'\bHuman rights\b', r'\bEthical concerns?\b',
    r'\bEnvironmental impact\b', r'\bPrivacy\b', r'\bData breach\b',
    r'\bSecurity (issues|concerns)\b', r'\bSafety concerns?\b', r'\bTerrorism\b',
    r'\bPropaganda\b'
]
_REGEXES = [re.compile(p, re.I) for p in _PATTERNS]

def contentious(title: str) -> bool:
    return any(rx.search(title or '') for rx in _REGEXES)

# --------------------------------------------------------------------------- #
# 2.  Safe recursive walker over the section tree
def walk_sections(sec: dict | None, parent: str = ''):
    if not isinstance(sec, dict):
        return
    name = (sec.get('name') or '').strip()
    full = f'{parent} / {name}' if parent else name

    def grab(node, buf):
        if isinstance(node, dict):
            if node.get('type') == 'paragraph':
                buf.append(node.get('value', ''))
            for ch in (node.get('has_parts') or []):
                grab(ch, buf)

    if contentious(name):
        bits = []; grab(sec, bits)
        yield full.lstrip(' /'), ' '.join(bits).strip()

    for ch in (sec.get('has_parts') or []):
        yield from walk_sections(ch, full)

# --------------------------------------------------------------------------- #
# 3.  **Special streaming helper for structured‑wikipedia**
def stream_structured_jsonl(source: str):
    """
    Yield raw article dicts from Wikimedia *structured‑wikipedia* shards.

    Parameters
    ----------
    source : str
        • If local==True  →  path to a directory that contains one or more
          `*.jsonl` or `*.jsonl.gz` files (all files are streamed, alphabetically).
        • If local==False →  subset id such as '20240916.en' (files pulled over HTTP).
    local : bool
        Switch between local‑disk mode and HTTP mode.
    max_ns : int
        Only used for remote mode: highest namespace number to check
        (enwiki_namespace_0.jsonl.gz … enwiki_namespace_<max_ns-1>.jsonl.gz).
    """
    # --- LOCAL DISK ------------------------------------------------------
    files = sorted(Path(source).glob("*.jsonl*"))
    if not files:
        raise FileNotFoundError(f"No *.jsonl* files found under {source!r}")
    for fp in files:
        try:
            ds = load_dataset(
                "json",
                data_files=str(fp),
                split="train",
                streaming=True,
                features=None,   # <-- no Arrow casting
            )
            for row in ds:
                yield row
        except Exception as e:
            logging.exception("Shard %s skipped (%s)", fp, e)

# --------------------------------------------------------------------------- #
# 4.  Main builder – chooses the correct iterator
def build_dataset(args):
    stream = stream_structured_jsonl(args.dataset_path)

    rows, n_good, n_bad = [], 0, 0
    for art in stream:
        if args.test_n and n_good >= args.test_n:
            break
        n_good += 1
        try:
            for top in art.get('sections') or []:
                for title, txt in walk_sections(top):
                    rows.append({
                        'article_title': art.get('name'),
                        'url': art.get('url'),
                        'section_title': title,
                        'section_text': txt
                    })
        except Exception:
            n_bad += 1
            logging.exception("Error processing article %s", art.get('name'))

        if n_good % 1000 == 0:
            print(f"Processed {n_good} articles")

    print(f"Scanned {n_good} articles  •  {n_bad} bad rows skipped"
          f"  •  kept {len(rows)} contentious sections")
    return Dataset.from_list(rows)

# --------------------------------------------------------------------------- #
# 5.  CLI wrapper
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', default=None, required=True)
    ap.add_argument('--test_n',  type=int, default=0,
                    help='Scan at most N good articles (0 = all)')
    ap.add_argument('--push_to_hub')
    ap.add_argument('--out_dir', default='contentious_sections')
    args = ap.parse_args()

    new_ds = build_dataset(args)
    if args.push_to_hub:
        new_ds.push_to_hub(args.push_to_hub, private=True)
    else:
        new_ds.save_to_disk(args.out_dir)

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    main()
