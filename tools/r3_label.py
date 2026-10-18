"""第三轮：用 Qwen3.5-4B-Base（nf4）给训练题打分。二次元、技术题的上文前加领域提示句（提示不参与评分），日常和维基不加。
用法：python tools/r3_label.py --data <run 目录> [--model Qwen/Qwen3.5-4B-Base] [--name q35_4b_base]"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = {"tech": "以下是软件开发与技术社区的讨论：", "anime": "以下是二次元社区的日常聊天："}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--name", default="q35_4b_base")
    a = ap.parse_args()
    data = Path(a.data)
    (data / "models_label.json").write_text(json.dumps({a.name: dict(hf=a.model, quant="nf4", role="teacher")}), encoding="utf-8")
    os.environ["DISTILL_DATA"] = str(data)
    os.environ["DISTILL_MODELS"] = str(data / "models_label.json")
    os.environ.setdefault("HF_HOME", "D:/distill-data/hf")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import common as C
    import s2_score
    C.install_signals()
    rows = [json.loads(l) for l in open(data / "items_train.jsonl", encoding="utf-8")]
    for r in rows:
        r["ctx"] = PROMPTS.get(r["domain"], "") + r["ctx"]
    C.write_jsonl(data / "items_train_prompted.jsonl", rows)
    s2_score.score_unit(f"{a.name}__train", data / "items_train_prompted.jsonl", lambda: s2_score.make_scorer(a.name))


if __name__ == "__main__":
    main()
