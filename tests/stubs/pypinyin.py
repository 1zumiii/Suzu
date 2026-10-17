"""自测用桩：字 → 拼音表。"""
PY = {}
def lazy_pinyin(w): return [PY.get(c, c) for c in w]
