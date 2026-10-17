"""总控：按顺序执行各阶段，每个阶段完成后留标记，重启后自动跳过已完成的。
退出码：0 全部结束；75 因停止请求（STOP 文件 / Ctrl-C）退出；1 出错（由 watchdog 重启）。
用法：python pipeline.py [--smoke] [--force-train] [--corpus 文件]
"""
import os
import sys
import traceback
from functools import partial

if "--smoke" in sys.argv:
    os.environ["DISTILL_SMOKE"] = "1"
if "--corpus" in sys.argv:
    os.environ["DISTILL_CORPUS"] = sys.argv[sys.argv.index("--corpus") + 1]

import common as C
from common import log

MAX_UNIT_FAILS = 4


def units():
    import s0_fetch, s1_build, s2_score, s3_report
    ms = C.models()
    teacher = os.environ.get("DISTILL_TEACHER") or next((k for k, v in ms.items() if v.get("role") == "teacher"), next(iter(ms)))
    if C.ROUND2:
        # 第二轮：评测题与基线打分沿用第一轮；新增四字评测，教师换成 4B 标注更多训练题
        u = [("fetch", s0_fetch.run, False), ("build", s1_build.run, False),
             ("score_ngram_four", partial(s2_score.run_eval, "ngram", "items_four", "__four"), False)]
        u += [(f"score_{k}_four", partial(s2_score.run_eval, k, "items_four", "__four"), False)
              for k in ms if k in ("q35_0p8b", "q35_4b")]
        u.append(("teacher_train", partial(s2_score.run_teacher_train, teacher), True))

        def train2():
            import s4_train
            s4_train.run()

        u += [("train", train2, True),
              ("score_student", partial(s2_score.run_eval, "student"), True),
              ("score_student_four", partial(s2_score.run_eval, "student", "items_four", "__four"), True),
              ("report_final", partial(s3_report.run, True), True)]
        return u
    u = [("fetch", s0_fetch.run, False), ("build", s1_build.run, False),
         ("score_ngram", partial(s2_score.run_eval, "ngram"), False)]
    u += [(f"score_{k}", partial(s2_score.run_eval, k), False) for k in ms]
    u.append(("report0", partial(s3_report.run, False), False))
    u.append(("teacher_train", partial(s2_score.run_teacher_train, teacher), True))

    def train():
        import s4_train
        s4_train.run()

    u.append(("train", train, True))
    u.append(("score_student", partial(s2_score.run_eval, "student"), True))
    u.append(("report_final", partial(s3_report.run, True), True))
    return u


def proceed(force):
    return force or bool((C.read_json(C.DATA / "gates.json") or {}).get("proceed")) or C.SMOKE


def main():
    C.install_signals()
    C.keep_awake()
    force = "--force-train" in sys.argv
    log(f"流水线启动（{'冒烟测试' if C.SMOKE else '正式'}，数据目录 {C.DATA}）")
    try:
        import preflight
        preflight.main()
    except Exception as e:
        log(f"预检跳过：{e}")
    for name, fn, gated in units():
        if C.is_finished(name):
            continue
        if C.stop_requested():
            return 75
        if gated and not proceed(force):
            C.mark(name, "skipped", "阶段 0 判定不继续")
            log(f"[{name}] 跳过（阶段 0 判定不继续蒸馏）")
            continue
        log(f"[{name}] 开始")
        try:
            fn()
        except C.StopRequested:
            log(f"[{name}] 已按请求停止，进度已保存")
            return 75
        except KeyboardInterrupt:
            return 75
        except Exception:
            log(f"[{name}] 出错：\n{traceback.format_exc()}")
            n = C.bump_fail(name)
            if n >= MAX_UNIT_FAILS:
                C.mark(name, "failed", f"连续失败 {n} 次")
                log(f"[{name}] 连续失败 {n} 次，标记为失败并继续后续阶段")
                continue
            return 1
        C.mark(name, "done")
        C.clear_fail(name)
        log(f"[{name}] 完成")
    C.atomic_write(C.BASE / "DONE.txt" if not C.SMOKE else C.DATA / "DONE.txt", "全部阶段结束，见 report.md\n")
    log("全部阶段结束")
    return 0


if __name__ == "__main__":
    sys.exit(main())
