"""公共设施：路径、配置、日志、停止信号、断点续跑用的 JSONL 和标记文件。只用标准库。"""
import ctypes
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SMOKE = os.environ.get("DISTILL_SMOKE") == "1"
BASE = Path(os.environ.get("DISTILL_DATA", ROOT / "data"))
DATA = BASE / "smoke" if SMOKE else BASE
STATE, SCORES, LOGS, CKPT = (DATA / n for n in ("state", "scores", "logs", "ckpt"))
for _p in (STATE, SCORES, LOGS, CKPT):
    _p.mkdir(parents=True, exist_ok=True)
STOP_FILE = BASE / "STOP"
os.environ.setdefault("HF_HOME", str(BASE / "hf"))   # 模型和语料缓存，可整体删除
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

CFG = dict(
    n_sentences=1_500_000,      # 语料句数（一个分片约能出 100 万+；不够就多下几个分片）
    per_article=6,              # 每篇文章最多取几句，避免被少数文章主导
    min_len=8, max_len=60,
    eval_mod=20,                # 句子哈希 % eval_mod == 0 的作为评测句（约 5%）
    min_count=3,                # 进入词表的最低词频
    k_cands=8,                  # 候选数上限（对应 Rime 一页候选）
    ctx_chars=30, min_ctx=2,
    max_per_truth=15,           # 同一个正确词最多出几题
    min_word_len=2,             # 出题的词最短字数：目标场景是两字词，单字词（的/是/在）会稀释指标
    eval_per_stratum=8000,      # 评测题：易例、难例各取这么多
    dev_per_stratum=1500,       # 开发集（选检查点用，与评测题的句子互不相交）
    train_scan=600_000,         # 训练题从多少个训练句里取
    train_items=120_000,
    ngram_train_sentences=400_000,
    score_seqs=24,              # 每次前向的序列数上限（显存不够会自动减半）
    student_steps=60_000, student_batch_lm=96, student_batch_kd=48,
    student_lr=6e-4, ckpt_every=1000, eval_every=2000,
    kd_weight=1.0, kd_temp=1.0,
    student_d=384, student_layers=8, student_heads=6, student_ctx=64, vocab_size=6000,
)
if os.environ.get("DISTILL_MIN_WORD_LEN"):
    CFG["min_word_len"] = int(os.environ["DISTILL_MIN_WORD_LEN"])
if SMOKE:
    CFG.update(n_sentences=4000, min_count=1, eval_per_stratum=150, dev_per_stratum=60, train_scan=3000,
               train_items=1500, ngram_train_sentences=3000, student_steps=150,
               ckpt_every=50, eval_every=75, student_batch_lm=32, student_batch_kd=16)

_stop = False


def install_signals():
    def handler(*_):
        global _stop
        _stop = True
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handler)


class StopRequested(Exception):
    pass


def stop_requested():
    return _stop or STOP_FILE.exists()


def check_stop():
    if stop_requested():
        raise StopRequested()


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    try:
        with open(LOGS / "pipeline.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def keep_awake():
    """阻止系统因空闲而睡眠（Windows）。必须在长期存活的线程里调用。"""
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def atomic_write(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=1))


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path, rows):
    atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def load_done_ids(path, key="id"):
    """读取已写出的记录 id。文件末尾被掐断的半行会被截掉，保证追加后仍是合法 JSONL。"""
    path = Path(path)
    if not path.exists():
        return set()
    data = path.read_bytes()
    end = data.rfind(b"\n") + 1
    if end != len(data):
        with open(path, "r+b") as f:
            f.truncate(end)
    ids = set()
    for line in data[:end].splitlines():
        try:
            ids.add(json.loads(line)[key])
        except (ValueError, KeyError):
            pass
    return ids


class Appender:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8", newline="\n")

    def write(self, obj):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def flush(self):
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.flush()
        self.f.close()


def marker(name, kind="done"):
    return STATE / f"{name}.{kind}"


def is_finished(name):
    return any(marker(name, k).exists() for k in ("done", "failed", "skipped"))


def mark(name, kind="done", info=""):
    marker(name, kind).write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S} {info}\n", encoding="utf-8")


def bump_fail(name):
    path = STATE / "failures.json"
    d = read_json(path, {})
    d[name] = d.get(name, 0) + 1
    write_json(path, d)
    return d[name]


def clear_fail(name):
    path = STATE / "failures.json"
    d = read_json(path, {})
    if d.pop(name, None) is not None:
        write_json(path, d)


def models():
    path = os.environ.get("DISTILL_MODELS", str(ROOT / "models.json"))
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if SMOKE:
        m = {k: v for k, v in m.items() if v.get("smoke")}
    return m


def md5_mod(s, n):
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) % n


def is_cjk(s):
    return bool(s) and all("一" <= c <= "鿿" for c in s)


class Progress:
    """每分钟写一行进度日志，并更新 state/progress.json 供 run.ps1 status 读取。"""

    def __init__(self, name, total, done=0):
        self.name, self.total, self.done0, self.done = name, total, done, done
        self.t0 = self.last = time.time()

    def tick(self, n=1, force=False):
        self.done += n
        now = time.time()
        if force or now - self.last >= 60:
            self.last = now
            rate = (self.done - self.done0) / max(now - self.t0, 1e-6)
            eta = (self.total - self.done) / rate if rate > 0 else 0
            msg = f"[{self.name}] {self.done}/{self.total}，{rate:.1f}/s，预计剩余 {eta/60:.0f} 分钟"
            log(msg)
            write_json(STATE / "progress.json", dict(unit=self.name, done=self.done, total=self.total,
                                                     rate=rate, eta_min=eta / 60, ts=time.time()))
