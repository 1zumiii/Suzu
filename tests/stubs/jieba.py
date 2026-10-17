"""自测用桩：最长匹配分词。"""
WORDS = set()
def setLogLevel(_): pass
def cut(s, HMM=False):
    i = 0
    while i < len(s):
        for n in (3, 2, 1):
            if s[i:i + n] in WORDS or n == 1:
                yield s[i:i + n]
                i += n
                break
