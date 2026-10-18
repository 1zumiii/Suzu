"""不依赖 torch/jieba/pypinyin/模型的端到端自测：玩具语料 + 桩 + dummy 打分。
覆盖：出题、n-gram、报告与判定、STOP 停止后续跑无重复、学生阶段缺 torch 时的失败隔离。"""
import json, os, random, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
tmp = Path(sys.argv[1])
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True)

# 玩具词表：同音词组
groups = {"de": "的地得", "shi": "是事时实", "zai": "在再", "zuo": "做作坐", "ta": "他她它", "yi": "以已已", "wo": "窝"}
py = {c: p for p, cs in groups.items() for c in cs}
py.update({"今": "jin", "天": "tian", "去": "qu", "公": "gong", "司": "si", "吃": "chi", "饭": "fan"})
(tmp / "stubs").mkdir()
(tmp / "stubs" / "jieba.py").write_text((HERE / "stubs" / "jieba.py").read_text())
(tmp / "stubs" / "pypinyin.py").write_text(f"PY={py!r}\ndef lazy_pinyin(w): return [PY.get(c, c) for c in w]\n")
rng = random.Random(3)
subj, verb, obj = "他她它", ["去公司", "吃饭", "做事", "坐在这里", "再做一次"], ["的时候", "是事实", "在公司", "作为"]
lines = [rng.choice(subj) + rng.choice("在再") + rng.choice(verb) + "，" + rng.choice(obj) + rng.choice("的地得") + str(rng.random())[2:4].translate(str.maketrans("0123456789", "今天去公司吃饭做事是")) + "作" * rng.randint(0, 3) for _ in range(6000)]
(tmp / "corpus.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
(tmp / "models.json").write_text(json.dumps({
    "small": {"backend": "dummy", "sigma": 3.0, "role": "small", "hf": "dummy/small"},
    "teacher": {"backend": "dummy", "sigma": 0.5, "role": "teacher", "hf": "dummy/teacher"}}))

env = dict(os.environ, PYTHONPATH=f"{tmp/'stubs'}{os.pathsep}{ROOT / 'pipeline'}", DISTILL_DATA=str(tmp / "data"),
           DISTILL_MODELS=str(tmp / "models.json"), DISTILL_CORPUS=str(tmp / "corpus.txt"), DISTILL_MIN_WORD_LEN="1")
cmd = [sys.executable, str(ROOT / "pipeline" / "pipeline.py")]
# 1) 带停止请求运行：dummy 慢速打分，中途放 STOP
env["DISTILL_SLOW"] = "0.2"
p = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
stop = tmp / "data" / "STOP"
t0 = time.time()
sc = tmp / "data" / "scores" / "small.jsonl"
while time.time() - t0 < 120 and not (sc.exists() and sc.stat().st_size > 2000):
    time.sleep(0.2)
stop.parent.mkdir(parents=True, exist_ok=True)
stop.write_text("")
out, _ = p.communicate(timeout=120)
print("第 1 次退出码", p.returncode, "（期望 75）")
assert p.returncode == 75, out[-2000:]
n1 = sum(1 for _ in open(sc))
stop.unlink()
# 2) 续跑直到结束（模拟 watchdog：出错码 1 则重启）
env.pop("DISTILL_SLOW")
for i in range(8):
    r = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
    print(f"续跑 {i+1} 退出码", r.returncode)
    if r.returncode == 0:
        break
else:
    raise SystemExit("未能结束\n" + r.stdout[-3000:])
ids = [json.loads(l)["id"] for l in open(sc)]
assert len(ids) == len(set(ids)), "分数文件出现重复题号"
items = [json.loads(l) for l in open(tmp / "data" / "items_eval.jsonl")]
assert len(ids) == len(items), (len(ids), len(items))
print(f"续跑后 small 共 {len(ids)} 题（停止时已有 {n1}），无重复")
st = tmp / "data" / "state"
print(sorted(x.name for x in st.iterdir()))
print((tmp / "data" / "report_stage0.md").read_text(encoding="utf-8"))
print((tmp / "data" / "gates.json").read_text())

# 3) 强制训练：本机没有 torch，学生阶段应连续失败 4 次后被隔离，流水线仍然收尾
for f in st.glob("*.skipped"): f.unlink()
for i in range(8):
    r = subprocess.run(cmd + ["--force-train"], env=env, cwd=ROOT, capture_output=True, text=True)
    print(f"强制训练 {i+1} 退出码", r.returncode)
    if r.returncode == 0: break
print(sorted(x.name for x in st.iterdir() if x.suffix in (".failed", ".skipped", ".done")))
assert (st / "train.failed").exists() and (tmp / "data" / "DONE.txt").exists()
print("自测通过")
