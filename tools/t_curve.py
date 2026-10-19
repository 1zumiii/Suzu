"""Dev-set learning curves from several run directories (logs/train_metrics.jsonl) as one markdown table.
Usage: python tools/t_curve.py --run NAME=<run dir> [--run ...] --out <md>"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = ["# Dev-set learning curves (natural-weighted accuracy / hard-example accuracy, percent)", ""]
    for spec in a.run:
        name, d = spec.split("=", 1)
        p = Path(d) / "logs" / "train_metrics.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        L += [f"## {name}", "", "| step | natural | hard | lm | kd |", "|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['step']} | {r['nat'] * 100:.1f} | {r['hard'] * 100:.1f} | {r['lm']:.3f} | {r['kd']:.3f} |")
        L.append("")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
