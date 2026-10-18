"""第三轮：用教师模型合成领域语料（日常口语、技术、二次元）。
每次请求给一个“话题/场景种子 + 语气”，让模型写 10 句；多次采样，温度 1.0。原始输出逐批追加到 <out>/raw_<domain>.jsonl（可断点续跑），
过滤与去重在 tools/filter_domain_corpus.py。提示词模板与生成参数见 DATA.md。
二次元领域只要求短的作品名、角色名、术语和日常讨论口吻，不引用作品原文、台词或歌词；不涉及任何真实个人数据。
用法：python tools/gen_domain_corpus.py --out <目录> --domain chat --target 5000 [--model Qwen/Qwen3.5-9B]"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TOPICS = {
    "chat": ["约饭", "周末安排", "加班吐槽", "点外卖", "奶茶", "天气变化", "通勤堵车", "租房搬家", "网购快递", "考试复习", "放假旅行",
             "看电影", "追剧", "打游戏", "健身减肥", "熬夜失眠", "感冒生病", "宠物猫狗", "家人电话", "朋友聚会", "迟到道歉", "借钱还钱",
             "手机没电", "找工作面试", "同事八卦", "领导安排", "生日礼物", "红包转账", "相亲聊天", "吵架和好", "做饭下厨", "逛超市",
             "买衣服", "理发", "上课点名", "宿舍室友", "毕业论文", "实习", "买车", "看病挂号", "取快递", "朋友圈", "表情包", "安慰朋友",
             "夸奖鼓励", "催促等人", "改签订票", "地铁公交", "下雨带伞", "深夜聊天"],
    "tech": ["代码评审", "线上故障排查", "数据库索引", "接口设计", "前端框架选型", "容器与部署", "持续集成", "版本管理与分支", "单元测试",
             "性能优化", "内存泄漏", "并发与锁", "缓存策略", "消息队列", "日志与监控", "权限与鉴权", "技术债务", "重构", "依赖升级",
             "编译报错", "类型系统", "异步编程", "网络协议", "负载均衡", "微服务拆分", "文档与注释", "开源项目贡献", "招聘与面试", "需求评审",
             "排期与延期", "代码风格", "命令行工具", "编辑器与插件", "机器学习训练", "模型推理部署", "数据清洗", "爬虫与反爬", "操作系统", "安全漏洞", "云服务费用"],
    "anime": ["新番讨论", "追番进度", "角色人设", "声优", "漫展与同人", "手游抽卡", "轻小说", "漫画更新", "剧场版", "原声音乐",
              "手办周边", "cosplay", "弹幕吐槽", "补番推荐", "神作与烂作", "百合与热血", "异世界题材", "校园恋爱番", "机战与科幻", "后宫与日常",
              "作画与导演", "OP和ED", "二创与MAD", "同好交流", "冷门作品安利", "老番怀旧", "游戏联动", "角色生日", "剧透与避雷", "谷子与吧唧"],
}
TONES = {
    "chat": ["随意口语", "语气词很多", "简短省略主语", "带点吐槽", "温柔关心", "开玩笑", "着急催促", "无奈抱怨"],
    "tech": ["同事间口头讨论", "论坛提问", "代码评审评论", "技术分享的叙述", "吐槽式", "严谨的说明"],
    "anime": ["圈内人聊天", "兴奋安利", "吐槽", "新手提问", "老粉怀旧", "弹幕风格"],
}
DOMAIN_DESC = {
    "chat": "朋友、同事、家人之间的日常聊天",
    "tech": "软件开发与技术社区里的交流",
    "anime": "二次元（动画、漫画、游戏）爱好者社区里的聊天",
}
TEMPLATE = ("请写 10 句简体中文，场景是{desc}，话题：{topic}，语气：{tone}。\n"
            "要求：每句 8 到 30 个字，口吻自然，像真人随手打出来的；每句内容各不相同，句式多样；"
            "不要编号，不要引号，不要解释，每行一句；不要出现真实的人名、电话、地址等个人信息；"
            "{extra}")
EXTRA = {
    "chat": "多用常见的日常词语。",
    "tech": "可以出现常见的技术术语（中英文混用时英文单词不超过一个）。",
    "anime": "可以提到作品名、角色名和圈内术语，但只写短的名称和你自己的评论，不要引用台词、歌词或剧情原文。",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", required=True, choices=list(TOPICS))
    ap.add_argument("--target", type=int, default=5000, help="目标原始句数（过滤前，约为最终的 1.3 倍时写这个数）")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quant", default="nf4")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(Path("D:/distill-data/hf")))
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    raw = out / f"raw_{a.domain}.jsonl"
    done_n = sum(1 for _ in open(raw, encoding="utf-8")) if raw.exists() else 0
    rng = random.Random(a.seed * 100003 + done_n)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    kw = {"device_map": {"": 0}}
    if a.quant == "nf4":
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, **kw)
    except Exception:
        cls = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForImageTextToText
        model = cls.from_pretrained(a.model, dtype=torch.bfloat16, **kw)
    model.eval()
    t0 = time.time()
    n_lines = done_n
    with open(raw, "a", encoding="utf-8") as f:
        while n_lines < a.target:
            metas, prompts = [], []
            for _ in range(a.batch):
                topic, tone = rng.choice(TOPICS[a.domain]), rng.choice(TONES[a.domain])
                p = TEMPLATE.format(desc=DOMAIN_DESC[a.domain], topic=topic, tone=tone, extra=EXTRA[a.domain])
                msgs = [{"role": "user", "content": p}]
                try:
                    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except TypeError:
                    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                metas.append((topic, tone))
                prompts.append(s)
            enc = tok(prompts, return_tensors="pt", padding=True).to(0)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=420, do_sample=True, temperature=a.temp, top_p=a.top_p, top_k=0,
                                   pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for (topic, tone), t in zip(metas, texts):
                for line in t.splitlines():
                    line = line.strip()
                    if line:
                        f.write(json.dumps({"d": a.domain, "topic": topic, "tone": tone, "t": line}, ensure_ascii=False) + "\n")
                        n_lines += 1
            f.flush()
            print(f"{a.domain}: {n_lines}/{a.target} 行，{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
