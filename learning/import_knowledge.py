"""
General knowledge from files YOU approve: the only way anything enters the "general" category.

    python -m learning.import_knowledge data/knowledge/my_facts.txt

The file must be in data/knowledge/ (putting it there is the approval). One statement per line, written by
you or copied from a human-written public-domain source; lines starting with # are notes. Each line is read
with the same rules as a taught sentence ("learn that ..."), stored with source = the file name and
confidence 0.85, and shown in the chat's Knowledge view. Nothing is ever imported automatically, and no AI
may write these files (the Originality Charter).
"""

import argparse
import sys
from pathlib import Path

from learning.knowledge import DEFAULT_DB
from learning.system import LearningSystem

ROOT = Path(__file__).resolve().parent.parent
APPROVED_DIR = ROOT / "data" / "knowledge"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Import approved general knowledge from data/knowledge/.")
    parser.add_argument("file")
    parser.add_argument("--knowledge-db", default=str(DEFAULT_DB))
    args = parser.parse_args(argv)
    path = Path(args.file).resolve()
    if APPROVED_DIR.resolve() not in path.parents:
        print(f"Refused: only files in {APPROVED_DIR.relative_to(ROOT)}/ are approved sources.")
        return 1
    system = LearningSystem(args.knowledge_db)
    try:
        added, skipped = system.import_file(path)
    finally:
        system.close()
    print(f"Imported {added} facts from {path.name} (general knowledge, source = {path.name}).")
    for line in skipped:
        print(f"  not understood, skipped: {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
