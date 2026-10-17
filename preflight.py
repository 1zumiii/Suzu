"""环境预检：显卡、依赖版本、磁盘、模型仓库是否存在。只报告，不修改任何东西。"""
import shutil
import sys

import common as C


def main():
    print("Python", sys.version.split()[0])
    try:
        import torch
        print("torch", torch.__version__, "CUDA", torch.version.cuda, "可用", torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"显卡 {p.name}，{p.total_memory/2**30:.1f} GB，算力 sm_{p.major}{p.minor}")
            if (p.major, p.minor) >= (12, 0) and tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2]) < (12, 8):
                print("警告：RTX 50 系需要 CUDA 12.8 及以上的 torch")
    except Exception as e:
        print("torch 不可用：", e)
    for m in ("transformers", "bitsandbytes", "jieba", "pypinyin", "opencc", "pyarrow", "huggingface_hub"):
        try:
            mod = __import__(m)
            print(m, getattr(mod, "__version__", "ok"))
        except Exception as e:
            print(m, "缺失：", type(e).__name__)
    free = shutil.disk_usage(C.BASE).free / 2**30
    print(f"数据目录 {C.BASE} 剩余 {free:.0f} GB（完整流程约需 45 GB）")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for k, v in C.models().items():
            try:
                api.model_info(v["hf"])
                print(f"模型 {v['hf']}：存在")
            except Exception as e:
                print(f"模型 {v['hf']}：查询失败（{type(e).__name__}），请核对 models.json 或网络")
    except Exception:
        pass


if __name__ == "__main__":
    main()
