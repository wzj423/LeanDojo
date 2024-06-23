import json
import time
from tqdm import tqdm, trange
import random
from copy import copy
from pathlib import Path
from loguru import logger

import lean_dojo
from lean_dojo import *
from typing import *


SPLIT_NAME = str  # train/val/test
SPLIT = Dict[SPLIT_NAME, List[TracedTheorem]]
SPLIT_STRATEGY = str
DST_DIR = Path("./exported_data")
NUM_VAL = NUM_TEST = 0

def _split_sequentially(
    traced_theorems: List[TracedTheorem],
) -> SPLIT:
    """Split ``traced_theorems`` sequentially into train/val/test."""
    num_theorems = len(traced_theorems)
    num_train = num_theorems #- NUM_VAL - NUM_TEST
    return {
        "train": traced_theorems, #[:num_train],
        # "val": traced_theorems[num_train : num_train + NUM_VAL],
        # "test": traced_theorems[num_train + NUM_VAL :],
    }


def split_randomly(
    traced_theorems: List[TracedTheorem],
) -> SPLIT:
    """Split ``traced_theorems`` randomly into train/val/test."""
    logger.info("Splitting the theorems randomly")
    # traced_theorems = copy(traced_theorems)
    # random.shuffle(traced_theorems)
    return _split_sequentially(traced_theorems)

from loguru import logger
def split_data(traced_repo: TracedRepo):
    # Skip theorems in the Lean 4 repo itself.
    traced_theorems = [
        thm for thm in traced_repo.get_traced_theorems() if not thm.repo.is_lean4
    ]
    logger.info(f"{len(traced_theorems)} theorems in total")

    return {
        "random": split_randomly(traced_theorems),
    }

def export_proofs(splits: Dict[SPLIT_STRATEGY, SPLIT], dst_path: Path, url) -> None:
    """Export all proofs in a traced repo to ``dst_path''."""
    for strategy, split in splits.items():
        split_dir = Path(dst_path) / strategy
        split_dir.mkdir(parents=True, exist_ok=True)

        for name, theorems in split.items():
            data = []
            data_term = []
            data_aesop, data_std = [], []
            num_tactics = 0

            for thm in theorems:
                tactics = [
                    {
                        # "small_step_tactic": t.get_traced_smallstep_tactics(),
                        "tactic": t.tactic,
                        "annotated_tactic": t.get_annotated_tactic(),
                        "state_before": t.state_before,
                        "state_after": t.state_after,
                    }
                    for t in thm.get_traced_tactics()
                    if t.state_before != "no goals"
                    and "·" not in t.tactic  # Ignore "·".
                ]
                num_tactics += len(tactics)
                entry = {
                        "url": thm.repo.url,
                        "commit": thm.repo.commit,
                        "file_path": str(thm.theorem.file_path),
                        "full_name": thm.theorem.full_name,
                        "start": list(thm.start),
                        "end": list(thm.end),
                        "traced_tactics": tactics,
                        "statement": thm.get_theorem_statement(),
                        "split": "test"
                    }
                if url != str(thm.repo.url) or str(thm.theorem.file_path).startswith("MiniF2F"):
                    continue

                elif thm.has_tactic_proof():
                    data.append(entry)
                else:
                    data_term.append(entry)

            oup_path = split_dir / f"{name}.json"
            json.dump(data, oup_path.open("wt"),ensure_ascii=False)
            logger.info(
                f"{len(data)} theorems saved to {oup_path}"
            )
            


if __name__ == '__main__':
    import argparse

    # Initialize the argparse.ArgumentParser
    parser = argparse.ArgumentParser(description='Process and export theorem proofs.')
    # Add input path argument
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help='The input path to the traced repo files.')
    # Add output path argument
    parser.add_argument('-o', '--output', type=str, required=True, 
                        help='The output path for the exported proofs.')
    # Parse the arguments
    args = parser.parse_args()
    # Call the main function with parsed arguments
    traced_repo = TracedRepo.from_traced_files(args.input, False)
    traced_theorems = split_data(traced_repo)
    export_proofs(traced_theorems,args.output,url=traced_repo.repo.url)