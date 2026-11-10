"""Source-only checkpoint selection, independent of target evaluation fields."""

import math


def select_source_iid(records):
    rows = list(records)
    if not rows:
        raise ValueError("Cannot select a checkpoint from empty records")
    for row in rows:
        if not math.isfinite(float(row["train_out"])):
            raise ValueError("Source validation accuracy is not finite")
    # max keeps the first maximum, matching the original Q.argmax tie rule.
    return max(rows, key=lambda row: row["train_out"])
