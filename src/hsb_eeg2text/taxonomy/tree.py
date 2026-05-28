from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hsb_eeg2text.utils.io import read_table


@dataclass(frozen=True)
class SemanticPath:
    coarse: str
    mid: str
    fine: str

    def as_list(self) -> list[str]:
        return [self.coarse, self.mid, self.fine]


class Taxonomy:
    def __init__(self, rows):
        self.rows = rows
        self.fine_to_path = {
            str(row["fine"]): SemanticPath(str(row["coarse"]), str(row["mid"]), str(row["fine"]))
            for _, row in rows.iterrows()
        }
        self.coarse_labels = sorted({p.coarse for p in self.fine_to_path.values()})
        self.mid_labels = sorted({p.mid for p in self.fine_to_path.values()})
        self.fine_labels = sorted(self.fine_to_path.keys())
        self.coarse_to_mids: dict[str, set[str]] = {}
        self.mid_to_fines: dict[str, set[str]] = {}
        for path in self.fine_to_path.values():
            self.coarse_to_mids.setdefault(path.coarse, set()).add(path.mid)
            self.mid_to_fines.setdefault(path.mid, set()).add(path.fine)

    @classmethod
    def from_csv(cls, path: str | Path) -> "Taxonomy":
        return cls(read_table(path))

    def path_for_fine(self, fine: str) -> SemanticPath:
        return self.fine_to_path[fine]

    def children_of_coarse(self, coarse: str) -> list[str]:
        return sorted(self.coarse_to_mids.get(coarse, set()))

    def children_of_mid(self, mid: str) -> list[str]:
        return sorted(self.mid_to_fines.get(mid, set()))

    def validate(self) -> dict:
        missing = []
        duplicate_fine = self.rows["fine"][self.rows["fine"].duplicated()].tolist()
        for col in ["keyword", "coarse", "mid", "fine"]:
            if col not in self.rows.columns:
                missing.append(col)
        return {
            "ok": not missing and not duplicate_fine,
            "missing_columns": missing,
            "duplicate_fine": duplicate_fine,
            "coarse_count": len(self.coarse_labels),
            "mid_count": len(self.mid_labels),
            "fine_count": len(self.fine_labels),
        }


def lca_depth(true_path: list[str], pred_path: list[str]) -> int:
    depth = 0
    for true_node, pred_node in zip(true_path, pred_path):
        if true_node != pred_node:
            break
        depth += 1
    return depth


def hierarchical_distance(true_path: list[str], pred_path: list[str]) -> int:
    depth = lca_depth(true_path, pred_path)
    return (len(true_path) - depth) + (len(pred_path) - depth)
