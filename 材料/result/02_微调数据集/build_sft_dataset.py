# -*- coding: utf-8 -*-
"""Qwen3-8B 语音导航意图识别 · 专用 SFT 数据集构建。

结论：`语音导航意图识别评测数据集.jsonl` 是冻结测试集，且标签覆盖仅 10.71%，
不能直接拿来微调。本脚本从「意图树 + 标注平台原始语料」重新构建一份
格式与评测严格对齐、与评测集零重叠的 SFT 数据集。

输入（只读，工作区根）：
  export_11.xlsx                        标注平台原始 2082 行
  意图树_完整.json                      498 个节点
  语音导航意图识别评测数据集.jsonl      432 条冻结测试集（仅用于防泄漏校验）
  build_dataset.py                      复用 xlsx 读取 / 清洗 / 评测 prompt 原文

输出（本目录）：
  sft_train.jsonl / sft_dev.jsonl / sft_test.jsonl   LLaMA-Factory / ms-swift 通用 messages 格式
  sft_labels.json                                    495 标签 + 498 路径
  sft_dataset_info.json                              规模统计 + 防泄漏校验 + sha256
  微调数据集说明.md                                  人读说明（由本脚本生成）

用法：
  python build_sft_dataset.py                        # 默认：严格隔离版，full 候选树
  python build_sft_dataset.py --tree-mode names     # 只给末级名称（无层级）
  python build_sft_dataset.py --tree-mode none      # 不给候选（closed-book）
  python build_sft_dataset.py --include-eval        # 额外产出含 432 条评测样本的 with_eval 版本
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(OUT))
sys.path.insert(0, ROOT)

from build_dataset import (  # noqa: E402
    read_xlsx, to_table, clean_text, skip_utterance, build_prompt, GENERIC_INTENTS,
)

F_EXPORT = os.path.join(ROOT, "export_11.xlsx")
F_TREE = os.path.join(ROOT, "意图树_完整.json")
F_EVAL = os.path.join(ROOT, "语音导航意图识别评测数据集.jsonl")

# 与评测 prompt 完全一致（v0）之外，再给两种等价改写，避免过拟合单一措辞。
# 三者都保持「单条 user message + 同样的四个字段 + 同样输出约束」。
INSTR_V0 = build_prompt  # 直接复用，保证与线上评测 100% 同构

HEAD_V1 = (
    '你是银行语音导航系统的意图识别模块。请阅读【候选意图树】、【历史对话】、'
    '【客户当前对话】和【小模型识别意图】，挑出唯一最匹配的意图节点。\n'
    '回答规范：\n'
    '1. 答案必须是候选意图树中出现的节点，禁止自造名称；\n'
    '2. 只写路径末级名称，不写完整路径、序号、标点或解释；\n'
    '3. 【小模型识别意图】只作参考，与对话冲突时以对话为准；\n'
    '4. 实在无法匹配时输出「集外」。\n\n'
)
HEAD_V2 = (
    '下面是一场银行语音导航对话。请结合上下文，从给定的候选意图树里选出最合适的一个意图。\n'
    '约束：\n'
    '1. 只能选树内节点，不能编造；\n'
    '2. 仅输出该节点层级路径的最后一段（意图名称），不要任何多余内容；\n'
    '3. 小模型识别意图仅供参考，请以客户实际表达为准；\n'
    '4. 无匹配项时输出「集外」。\n\n'
)

# 口语化客户问法模板（用于业务/FAQ 节点；节点名称本身即意图短语）
Q_TEMPLATES = [
    "{n}",
    "我想咨询一下{n}",
    "{n}怎么办",
    "帮我查一下{n}",
    "请问{n}是怎么规定的",
    "{n}，能办吗",
    "我要办{n}",
    "{n}这个业务",
]

# 通用交互意图的真实口语（这些节点用业务模板造句不合理，单独给）
GENERIC_UTTERANCES = {
    "人工服务": ["转人工", "我要找人工", "人工客服", "接人工", "帮我转人工服务"],
    "友好问候": ["你好", "您好", "喂", "在吗"],
    "肯定": ["对", "是的", "嗯", "好", "没错", "可以"],
    "否定": ["不是", "不对", "不要", "没有", "不办了"],
    "简短词": ["嗯", "啊", "哦", "那个", "额"],
    "结束语": ["没了", "就这样", "不咨询了", "谢谢，再见"],
    "帮助": ["你能做什么", "有什么业务", "帮我看一下菜单", "怎么用"],
    "业务列表": ["有哪些业务", "业务菜单", "都能办什么"],
    "业务咨询": ["我想咨询个业务", "有个问题想问", "想了解一下"],
    "投诉建议": ["我要投诉", "提个建议", "你们这个服务太差了"],
    "抱怨": ["太慢了", "怎么又这样", "听不清", "烦死了"],
    "答案播报": ["再说一遍", "没听清", "重复一下", "刚才说什么"],
    "老年服务": ["老年服务", "我听不太清", "说慢一点"],
    "NULL": [],
    "无效意图": [],
    "集外": [],  # 单独用域外问法生成
    "中银富登": ["中银富登", "富登银行的业务"],
    "本地特有服务": ["本地特色业务", "本地服务"],
}

# 明确的域外问法（用于训练「集外」拒识，全部为合成）
OOD_UTTERANCES = [
    "今天天气怎么样", "帮我订一份外卖", "讲个笑话", "股票今天涨了吗",
    "附近有什么电影院", "明天几点下雨", "帮我写一首诗", "翻译一句英文",
    "我要买一张火车票", "快递到哪了", "播放一首歌", "今天的日期是多少",
    "怎么做红烧肉", "帮我算一下 37 乘以 89", "给我讲讲量子力学",
    "附近的健身房在哪", "我想预约打车", "这个手机怎么重启", "推荐一部电影",
    "你会说英语吗",
]


def load_inputs():
    exp = to_table(read_xlsx(F_EXPORT))
    tree = json.load(open(F_TREE, "r", encoding="utf-8"))["nodes"]
    with open(F_EVAL, "r", encoding="utf-8") as f:
        ev = [json.loads(x) for x in f if x.strip()]
    return exp, tree, ev


def make_user_prompt(instr_variant, tree_text, n_nodes, history, utterance, small):
    head = HEAD_V1 if instr_variant == 1 else HEAD_V2
    tree_block = ("（共 %d 个候选意图，每行一个名称）\n%s" % (n_nodes, tree_text)
                  if tree_text else "（本次不提供候选清单，请依据已知意图体系作答）")
    return (
        head +
        "【候选意图树】%s\n\n" % tree_block +
        "【历史对话】\n%s\n\n" % (history or "（无历史对话）") +
        "【客户当前对话】\n%s\n\n" % utterance +
        "【小模型识别意图】\n%s\n\n" % (small or "（空）") +
        "【输出】\n"
    )


def split_of(group_key: str) -> str:
    """按组稳定切分 8:1:1，同组绝不跨集合（md5，不受 PYTHONHASHSEED 影响）。"""
    h = int(hashlib.md5(group_key.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if h < 8 else ("dev" if h == 8 else "test")


def build(args):
    exp, tree, eval_rows = load_inputs()
    rng = random.Random(args.seed)

    paths = [n["path"] for n in tree]
    path_set = set(paths)
    labels = sorted({p.split("/")[-1] for p in paths})
    label_set = set(labels)
    tree_text_full = "\n".join(paths)
    tree_text_names = "\n".join(labels)
    tree_cfg = {
        "full": (tree_text_full, len(paths)),
        "names": (tree_text_names, len(labels)),
        "none": ("", 0),
    }
    tree_text, n_nodes = tree_cfg[args.tree_mode]

    eval_rows_by_id = {int(r["source_row"]): r for r in eval_rows}
    eval_session_set = {str(r["session_id"]) for r in eval_rows}
    eval_utt_set = {str(r["current_utterance"]).strip() for r in eval_rows}

    # 会话分组（用于构造历史 + 真实样本）
    sessions = collections.defaultdict(list)
    for excel_row, r in enumerate(exp, start=2):
        r["_excel_row"] = excel_row
        sessions[r["会话ID"].strip()].append(r)
    for sid in sessions:
        sessions[sid].sort(key=lambda x: int(x["轮次序号"])
                          if x["轮次序号"].strip().isdigit() else -1)

    # 客服话术池：给合成样本补一点真实历史上下文
    agent_pool = []
    for r in exp:
        if r["说话人"].strip() != "agent":
            continue
        t = clean_text(r["语句文本"])
        if 10 <= len(t) <= 160 and not skip_utterance(t):
            agent_pool.append(t)
    agent_pool = list(dict.fromkeys(agent_pool))

    def random_history():
        if not agent_pool or rng.random() < 0.55:
            return ""
        k = 1 if rng.random() < 0.8 else 2
        return "\n".join("客服：%s" % x for x in rng.sample(agent_pool, min(k, len(agent_pool))))

    def random_small(label):
        p = rng.random()
        if p < 0.55:
            return label
        if p < 0.80:
            return "未识别"
        return rng.choice(labels)  # 小模型给错，训练模型「以对话为准」

    records = []

    def add(utterance, label, *, source_type, group_key, node_path="",
            history="", small="", meta=None):
        if label not in label_set:
            return False
        # 与冻结评测集任一「客户当前对话」原文完全相同的问法直接丢弃（含极短口语）
        if str(utterance).strip() in eval_utt_set:
            return False
        v = rng.randrange(3)
        if v == 0:
            prompt = INSTR_V0(tree_text, n_nodes, history, utterance, small)
        else:
            prompt = make_user_prompt(v, tree_text, n_nodes, history, utterance, small)
        d = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": label},
            ],
            "metadata": {
                "sample_id": "T%05d" % (len(records) + 1),
                "source_type": source_type,
                "group_key": group_key,
                "label": label,
                "node_path": node_path,
                "instruction_variant": "v%d" % v,
                "prompt_chars": len(prompt),
                "has_history": bool(history),
                "small_model_intent": small,
                "split": split_of(group_key),
            },
        }
        if meta:
            d["metadata"].update(meta)
        records.append(d)
        return True

    # ---- 1) 真实标注样本：严格排除评测集源行及其同会话行 ----
    n_real = 0
    real_labels = set()
    for sid in sorted(sessions, key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
        rows = sessions[sid]
        for r in rows:
            if r["说话人"].strip() != "user" or r["_excel_row"] in eval_rows_by_id:
                continue
            if str(r["会话ID"]).strip() in eval_session_set:
                continue  # 会话级隔离：评测集出现过的会话整段不进训练
            label = r["人工标注意图名称"].strip()
            if not label or label.upper() == "NULL" or label not in label_set:
                continue
            reason = skip_utterance(r["语句文本"])
            if reason:
                continue
            cur_turn = int(r["轮次序号"]) if r["轮次序号"].strip().isdigit() else 0
            prev = [x for x in rows if x["轮次序号"].strip().isdigit()
                    and int(x["轮次序号"]) < cur_turn][-5:]
            hist = "\n".join(
                "%s：%s" % ("客户" if x["说话人"].strip() == "user" else "客服",
                           clean_text(x["语句文本"]))
                for x in prev if clean_text(x["语句文本"]))
            ok = add(r["语句文本"].strip(), label, source_type="real_labeled",
                     group_key="real:%s" % sid, node_path=r["人工标注意图"].strip(),
                     history=hist, small=r["机器识别意图"].strip(),
                     meta={"session_id": sid, "turn_index": cur_turn,
                           "source_row": r["_excel_row"]})
            if ok:
                n_real += 1
                real_labels.add(label)

    # ---- 2) 真实 ASR 噪声 / 无效行：训练「简短词 / 集外」 ----
    n_noise = 0
    for sid in sorted(sessions, key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
        for r in sessions[sid]:
            if r["说话人"].strip() != "user" or r["_excel_row"] in eval_rows_by_id:
                continue
            reason = skip_utterance(r["语句文本"])
            if reason not in ("英文", "单个数字", "单个符号"):
                continue
            label = "简短词" if reason in ("单个数字", "单个符号") else "集外"
            if label not in label_set:
                continue
            add(r["语句文本"].strip(), label, source_type="real_noise",
                group_key="noise:%d" % r["_excel_row"],
                history="", small="未识别",
                meta={"session_id": sid, "source_row": r["_excel_row"],
                      "noise_type": reason})
            n_noise += 1
            if n_noise >= args.max_noise:
                break
        if n_noise >= args.max_noise:
            break

    # ---- 3) 意图树驱动的冷启动样本（覆盖 498 个节点） ----
    n_synth = 0
    n_oral = 0
    for n in tree:
        path = n["path"]
        label = path.split("/")[-1]
        group = "synth:%s" % path
        parts = path.split("/")
        if label in GENERIC_UTTERANCES or label in GENERIC_INTENTS:
            utts = GENERIC_UTTERANCES.get(label) or []
            if not utts:
                continue  # 集外 / NULL / 无效意图 单独处理
            for u in rng.sample(utts, min(args.per_node, len(utts))):
                add(u, label, source_type="synthetic_oral", group_key=group,
                    node_path=path, history=random_history(), small=random_small(label),
                    meta={"level": n["level"], "source_file": "/".join(n["sources"])})
                n_oral += 1
        else:
            names = [label]
            if len(parts) > 1:
                names.append("%s，%s" % (parts[-2], label))
            tids = rng.sample(range(len(Q_TEMPLATES)), args.per_node)
            for k, tid in enumerate(tids):
                base = names[k % len(names)]
                u = Q_TEMPLATES[tid].format(n=base)
                add(u, label, source_type="synthetic_template", group_key=group,
                    node_path=path, history=random_history(), small=random_small(label),
                    meta={"level": n["level"], "source_file": "/".join(n["sources"]),
                          "template_id": tid})
                n_synth += 1

    # ---- 4) 域外拒识样本 ----
    n_ood = 0
    for i, u in enumerate(OOD_UTTERANCES):
        if "集外" not in label_set:
            break
        add(u, "集外", source_type="synthetic_ood", group_key="ood:%d" % i,
            node_path="集外", history=random_history(), small=rng.choice(["未识别", "人工服务"]))
        n_ood += 1

    # ---- 5) 可选：把 432 条评测样本并入（会污染冻结测试集） ----
    with_eval = []
    if args.include_eval:
        for r in eval_rows:
            label = str(r["ground_truth"]).strip()
            if label not in label_set:
                continue
            v = 0  # 与评测完全一致
            prompt = INSTR_V0(tree_text, n_nodes,
                              str(r.get("history_text") or ""),
                              str(r.get("current_utterance") or ""),
                              str(r.get("small_model_intent") or ""))
            with_eval.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": label},
                ],
                "metadata": {
                    "sample_id": r["sample_id"], "source_type": "eval_included",
                    "group_key": "eval:%s" % r["sample_id"], "label": label,
                    "node_path": r.get("human_intent_path") or "",
                    "instruction_variant": "v0", "prompt_chars": len(prompt),
                    "source_row": r["source_row"], "split": "train",
                },
            })

    # ---- 切分与校验 ----
    buckets = {"train": [], "dev": [], "test": []}
    for d in records:
        buckets[d["metadata"]["split"]].append(d)
    group_split = collections.defaultdict(set)
    for d in records:
        group_split[d["metadata"]["group_key"]].add(d["metadata"]["split"])
    cross_group = [g for g, s in group_split.items() if len(s) > 1]

    eval_src = set(eval_rows_by_id)
    leaked_src = [d["metadata"].get("source_row") for d in records
                  if d["metadata"].get("source_row") in eval_src]
    # 精确校验：抽出每条训练样本的「客户当前对话」，与冻结评测集原文做集合比对（O(n)）
    leaked_utt = []
    re_cur = re.compile(r"【客户当前对话】\n(.*?)\n\n【小模型识别意图】", re.S)
    for d in records:
        m = re_cur.search(d["messages"][0]["content"])
        if m and m.group(1).strip() in eval_utt_set:
            leaked_utt.append(m.group(1).strip())

    checks = {
        "all_labels_in_tree": all(d["messages"][1]["content"] in label_set for d in records),
        "messages_shape_ok": all(len(d["messages"]) == 2
                                 and d["messages"][0]["role"] == "user"
                                 and d["messages"][1]["role"] == "assistant"
                                 and "\n" not in d["messages"][1]["content"]
                                 for d in records),
        "no_eval_source_row": not leaked_src,
        "no_eval_utterance": not leaked_utt,
        "no_group_cross_split": not cross_group,
        "no_empty_utterance": all("【客户当前对话】\n\n" not in d["messages"][0]["content"]
                                  for d in records),
    }

    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for d in items:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    for split, items in buckets.items():
        write_jsonl(os.path.join(OUT, "sft_%s.jsonl" % split), items)
    if args.include_eval:
        # dev / test 与默认版完全相同，不重复落盘，避免产生歧义文件
        write_jsonl(os.path.join(OUT, "sft_train_with_eval.jsonl"),
                    buckets["train"] + with_eval)

    with open(os.path.join(OUT, "sft_labels.json"), "w", encoding="utf-8") as f:
        json.dump({"n_paths": len(paths), "n_labels": len(labels),
                   "paths": paths, "labels": labels}, f, ensure_ascii=False, indent=2)

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    n_unlabeled_invalid = sum(1 for r in exp
                              if r["说话人"].strip() == "user"
                              and r["是否无效意图"].strip() == "是"
                              and not r["人工标注意图名称"].strip())
    # 仅按「源行不在评测集」过滤时的剩余有效标注行数（未做会话隔离），用于说明隔离代价
    remaining_valid_labeled_rows = sum(
        1 for i, r in enumerate(exp, start=2)
        if r["说话人"].strip() == "user" and i not in eval_rows_by_id
        and r["人工标注意图名称"].strip()
        and r["人工标注意图名称"].strip().upper() != "NULL")
    src_stat = collections.Counter(d["metadata"]["source_type"] for d in records)
    split_stat = {k: len(v) for k, v in buckets.items()}
    label_cov = {k: len({d["metadata"]["label"] for d in v}) for k, v in buckets.items()}
    covered_paths = {d["metadata"]["node_path"] for d in records
                     if d["metadata"]["node_path"] in path_set}
    node_cov = len(covered_paths)
    uncovered_nodes = sorted(path_set - covered_paths)
    avg_chars = sum(d["metadata"]["prompt_chars"] for d in records) / max(len(records), 1)

    info = {
        "generator": "build_sft_dataset.py",
        "seed": args.seed,
        "tree_mode": args.tree_mode,
        "per_node": args.per_node,
        "include_eval_generated": bool(args.include_eval),
        "eval_included_in_default_files": False,
        "n_records": len(records),
        "splits": split_stat,
        "label_coverage_per_split": label_cov,
        "covered_tree_nodes": node_cov,
        "uncovered_tree_nodes": uncovered_nodes,
        "total_tree_nodes": len(paths),
        "node_coverage_pct": round(100.0 * node_cov / len(paths), 2),
        "source_type_counts": dict(src_stat),
        "real_labeled_samples": src_stat.get("real_labeled", 0),
        "real_labeled_labels": len(real_labels),
        "real_noise_samples": src_stat.get("real_noise", 0),
        "unlabeled_invalid_user_rows": n_unlabeled_invalid,
        "remaining_valid_labeled_rows_before_session_isolation": remaining_valid_labeled_rows,
        "synthetic_template_samples": src_stat.get("synthetic_template", 0),
        "synthetic_oral_samples": src_stat.get("synthetic_oral", 0),
        "synthetic_ood_samples": src_stat.get("synthetic_ood", 0),
        "avg_prompt_chars": round(avg_chars, 1),
        "checks": checks,
        "checks_all_pass": all(checks.values()),
        "sha256": {("sft_%s.jsonl" % k): sha(os.path.join(OUT, "sft_%s.jsonl" % k))
                   for k in buckets},
        "eval_source_sha256": sha(F_EVAL),
        "note": "严格隔离版：不含冻结评测集任何样本，也不含其出现过的会话。",
    }
    with open(os.path.join(OUT, "sft_dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    write_readme(info, label_set, args)

    print("总样本 %d ｜ train/dev/test = %d/%d/%d" %
          (len(records), split_stat["train"], split_stat["dev"], split_stat["test"]))
    print("来源：" + "，".join("%s=%d" % (k, v) for k, v in src_stat.most_common()))
    print("节点覆盖 %d/%d（%.2f%%）｜ 平均 prompt %.0f 字符" %
          (node_cov, len(paths), info["node_coverage_pct"], avg_chars))
    print("校验：" + ("全部 PASS" if info["checks_all_pass"] else "存在 FAIL -> %s" % checks))
    if not info["checks_all_pass"]:
        return 1
    return 0


def write_readme(info, label_set, args):
    p = os.path.join(OUT, "微调数据集说明.md")
    s = info["splits"]
    cov = info["label_coverage_per_split"]
    src = info["source_type_counts"]
    L = []
    A = L.append
    A("# Qwen3-8B 语音导航意图识别 · 专用微调（SFT）数据集说明")
    A("")
    A("> 生成脚本：`build_sft_dataset.py`；统计与校验见 `sft_dataset_info.json`。"
      "本 md 由脚本自动生成，重跑即刷新数字。")
    A("")
    A("## 一、结论：原来的评测集 jsonl **不能直接用于微调**")
    A("")
    A("| # | 问题 | 具体证据 |")
    A("| :-- | :--- | :--- |")
    A("| 1 | **测试集泄漏** | 这 432 条是冻结的专业能力测试集，`datasets/intent_routing/dataset_info.json` "
      "已记录源 sha256。拿它训练，之后用同一套评测代码测出的准确率完全失效。 |")
    A("| 2 | **标签覆盖极低** | 只命中 53/495 个末级标签（10.71%）、53/498 个节点（10.64%），"
      "445 个节点零命中；`人工服务` 独占 164 条（37.96%）。8B 模型微调后会强烈偏向高频通用意图。 |")
    A("| 3 | **prompt 形态是评测专用** | 每条把 498 条候选树整段内联，平均 6369 字符（约 3.2k tokens）。"
      "直接当训练语料，单次训练成本极高，且模型会学成「依赖超长候选列表做检索」。 |")
    A("| 4 | **不是 SFT 格式** | 每条是 `{sample_id, prompt, ground_truth, ...}` 的评测记录，"
      "没有 `messages` / `instruction-output` 结构，LLaMA-Factory、ms-swift 无法直接读取。 |")
    A("| 5 | **只有单一正样本、单一模板** | 432 条共用同一段指令、同一份候选树；"
      "没有指令多样性、没有「小模型给错要纠正」的样本、没有域外拒识与 ASR 噪声样本。 |")
    A("| 6 | **真实语料几乎被抽空** | `export_11.xlsx` 共 1041 条 user 行，其中 602 条标注为空或 NULL、"
      "432 条已进评测集；仅按源行排除后只剩 %d 条有效标注，且这些行全部属于评测集出现过的会话，"
      "按会话级隔离后**训练集中真实标注正样本为 %d 条**（%d 个标签）。 |" %
      (info["remaining_valid_labeled_rows_before_session_isolation"],
       info["real_labeled_samples"], info["real_labeled_labels"]))
    A("")
    A("> ⚠️ **重要前提**：严格隔离版（`sft_train.jsonl`）里**没有任何一条带真实人工意图标注的正样本**，"
      "正样本全部由意图树合成。它的定位是「格式对齐 + 495 个标签的冷启动」，"
      "**不能单独支撑上线**。真实标注只有两条获取途径："
      "`--include-eval`（会污染测试集，见第四节）或补标（见第六节）。")
    A("")
    A("因此：另建一份**格式对齐、标签全覆盖、与评测集零重叠**的 SFT 数据集；"
      "同时必须说明其中合成样本的定位（见第六节）。")
    A("")
    A("## 二、数据集构成")
    A("")
    A("| 来源类型 | 条数 | 说明 |")
    A("| :--- | ---: | :--- |")
    A("| `real_labeled` | %d | export 中未被评测集使用、且**整段会话不在评测集**的真实标注语句，"
      "带真实历史对话与机器识别意图 |" % src.get("real_labeled", 0))
    A("| `real_noise` | %d | export 中真实的纯英文系统词 / 单数字 / 单符号噪声行，"
      "标为 `集外` 或 `简短词`，是最有价值的真实负样本 |" % src.get("real_noise", 0))
    A("| `synthetic_template` | %d | 按意图树节点名称用 %d 种口语句式生成，"
      "二三级节点额外带父级上下文；用于让模型**认识全部标签空间** |" %
      (src.get("synthetic_template", 0), len(Q_TEMPLATES)))
    A("| `synthetic_oral` | %d | 「人工服务 / 问候 / 肯定 / 否定 / 结束语」等通用交互意图，"
      "用真实口语短句生成（业务模板造句不合理） |" % src.get("synthetic_oral", 0))
    A("| `synthetic_ood` | %d | 明确的域外问法（天气、外卖、写诗等），标签 `集外`，训练拒识 |" %
      src.get("synthetic_ood", 0))
    A("")
    A("**总计 %d 条**；候选树模式 `%s`（`full` = 与评测一致的 498 条层级路径）。"
      "平均 prompt %.0f 字符；覆盖 %d/%d 个意图节点（%.2f%%）。" %
      (info["n_records"], args.tree_mode, info["avg_prompt_chars"],
       info["covered_tree_nodes"], info["total_tree_nodes"], info["node_coverage_pct"]))
    A("")
    A("未覆盖节点：%s。这些是评测阶段已剔除的字面占位标签，不应作为有效意图输出，"
      "因此刻意不生成训练样本。" %
      ("、".join("`%s`" % x for x in info["uncovered_tree_nodes"]) or "无"))
    A("## 三、格式与对齐口径（关键）")
    A("")
    A("每行一条 JSON，采用 LLaMA-Factory / ms-swift 通用的 `messages` 格式：")
    A("")
    A("```json")
    A('{"messages": [')
    A('  {"role": "user", "content": "你是银行语音导航系统的意图识别专家。…【候选意图树】…【历史对话】…【客户当前对话】…【小模型识别意图】…【输出】"},')
    A('  {"role": "assistant", "content": "个人账户交易明细查询"}')
    A('], "metadata": {"sample_id": "T00001", "source_type": "synthetic_template", "group_key": "synth:…",')
    A('                 "label": "个人账户交易明细查询", "node_path": "…", "split": "train", …}}')
    A("```")
    A("")
    A("**只有一条 `user` + 一条 `assistant`，刻意不加 `system` 角色。** 原因："
      "`geval/professional.py::IntentRoutingTask.build_messages()` 明确返回 "
      "`[{\"role\":\"user\",\"content\": question}]`，评测时框架只发单条 user message。"
      "若 SFT 训练成 system+user，微调后的模型在评测 prompt 上会分布不匹配。")
    A("")
    A("- `instruction_variant`：`v0` 直接复用 `build_dataset.build_prompt()` 原文（与评测 100% 同构），"
      "`v1/v2` 为等价措辞改写，防止过拟合单一指令。")
    A("- `assistant` 只输出**末级意图名称**，与 `IntentRoutingTask.extract_intent()` 的判分对象一致；"
      "不输出路径、不加解释、不换行。")
    A("- `small_model_intent` 刻意混入正确标签 / `未识别` / 随机错标签，"
      "对应 prompt 第 3 条「小模型意图仅供参考」，训练模型纠正小模型而非盲从。")
    A("- 节点描述 `description_clean` **没有进训练输入**：那是客服的应答话术，"
      "把答案当用户输入会制造错误的监督信号；只保留在 metadata 的 `node_path` 供追溯。")
    A("")
    A("## 四、切分与防泄漏")
    A("")
    A("| 集合 | 文件 | 条数 | 覆盖标签数 |")
    A("| :--- | :--- | ---: | ---: |")
    for k in ("train", "dev", "test"):
        A("| %s | `sft_%s.jsonl` | %d | %d |" % (k, k, s[k], cov[k]))
    A("")
    A("切分用 `md5(group_key) % 10` 稳定映射为 8:1:1，`group_key` 为 "
      "`real:<会话ID>` / `synth:<层级路径>` / `noise:<源行号>` / `ood:<序号>`，"
      "**同一组绝不跨集合**（同一节点的所有问法变体、同一会话的所有轮次不会被拆开）。")
    A("")
    A("自动校验（`sft_dataset_info.json` 的 `checks`）：")
    A("")
    A("| 校验项 | 结果 |")
    A("| :--- | :--- |")
    for k, v in info["checks"].items():
        A("| `%s` | %s |" % (k, "PASS" if v else "FAIL"))
    A("")
    A("- 默认文件**不含**冻结评测集任何 `source_row`、任何 `current_utterance` 原文；"
      "评测集出现过的 140 个会话，整段会话都不进训练集。")
    A("")
    A("### 4.1 可选变体 `--include-eval`：唯一能立刻拿到真实标注的方式")
    A("")
    A("`python build_sft_dataset.py --include-eval` 会额外产出单一的 "
      "`sft_train_with_eval.jsonl`（= 默认 train %d 条 + 432 条真实人工标注样本），"
      "dev / test 沿用默认文件、不重复落盘。新增样本指令用与评测完全一致的 `v0`，避免格式漂移。"
      "这是当前唯一能立刻拿到真实标注正样本的产物。" % s["train"])
    A("")
    A("**代价（必须接受）**：原 `语音导航意图识别评测数据集.jsonl` 从此成为训练集的一部分，"
      "继续用它做专业能力评测 = 训练集自测，准确率虚高、结论无效。选用这版必须：")
    A("")
    A("1. 另建一套**独立、不与任何训练样本同会话**的测试集（优先按 445 个零命中节点补标）；")
    A("2. 用 `scripts/prepare_professional_dataset.py --input <新测试集> "
      "--out datasets/intent_routing_v2` 转换成框架格式，评测配置指向新数据目录；")
    A("3. 所有对外结论只基于新测试集，旧 432 条只作为训练拟合情况参考。")
    A("")
    A("默认三个文件不含这 432 条；`sft_dataset_info.json` 的 "
      "`eval_included_in_default_files=false` 即为凭据。"
      "上面 `checks` 也只针对默认严格隔离版，不代表 with_eval 版未污染。")
    A("")
    A("## 五、Qwen3-8B 微调建议")
    A("")
    A("**框架**：LLaMA-Factory 或 ms-swift，LoRA / QLoRA。建议配置：")
    A("")
    A("```yaml")
    A("# LLaMA-Factory 训练参数（示例，按显存调整）")
    A("model_name_or_path: Qwen/Qwen3-8B")
    A("stage: sft")
    A("finetuning_type: lora")
    A("lora_rank: 32            # 标签空间 495 类，r=16 偏小")
    A("lora_alpha: 64")
    A("lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    A("learning_rate: 1.0e-4")
    A("num_train_epochs: 2      # 合成样本多，>3 轮极易过拟合模板")
    A("per_device_train_batch_size: 1")
    A("gradient_accumulation_steps: 8")
    A("cutoff_len: 4096         # full 模式 prompt 约 3.2k tokens，必须 >=4096")
    A("template: qwen3")
    A("flash_attn: fa2")
    A("```")
    A("")
    A("**Qwen3 特有的两个坑**：")
    A("")
    A("1. **关闭 thinking**：Qwen3 默认开启推理模式，会把 64 token 的输出预算吃光并输出一大段分析。"
      "训练数据里 assistant 只有标签，需在推理侧传 "
      "`extra_body.chat_template_kwargs.enable_thinking=false`（评测配置已带）。")
    A("2. **不要混用 system 角色**：见第三节；`template: qwen3` 会把 messages 渲染成 "
      "`<|im_start|>user … <|im_start|>assistant …`，与评测端一致。")
    A("")
    A("`dataset_info.json` 片段（LLaMA-Factory）：")
    A("")
    A("```json")
    A('{"voice_nav_sft": {"file_name": "sft_train.jsonl",')
    A('  "formatting": "sharegpt",')
    A('  "columns": {"messages": "messages"},')
    A('  "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}}')
    A("```")
    A("")
    A("训练完成后用 `../03_Qwen3-8B评测/Qwen3-8B专业能力评测说明.md` 里的命令，"
      "把端点指到微调后的模型，与微调前用不同 `--run-name` 做 A/B。")
    A("")
    A("## 六、局限（务必读完）")
    A("")
    A("- **这是「格式对齐 + 标签空间冷启动」数据集，不是生产级微调语料。** "
      "合成样本占 %.1f%%，真实标注仅 %d 条；问法由节点名称套句式生成，"
      "与真实 ASR 口语（含语气词、断句、识别错误）差距明显。" %
      (100.0 * (info["n_records"] - src.get("real_labeled", 0) - src.get("real_noise", 0))
       / max(info["n_records"], 1), info["real_labeled_samples"]))
    A("- **覆盖率 ≠ 质量**：%d/%d 个节点都出现过，但每个节点只有 %d 条左右，"
      "且是同义改写，不足以学会细粒度区分（例如「借记卡换卡」vs「借记卡挂失换卡」）。" %
      (info["covered_tree_nodes"], info["total_tree_nodes"], args.per_node))
    A("- **内部 dev/test 只验证格式与标签记忆**，不能用来宣称模型线上效果；"
      "真正的效果指标只能用冻结的 432 条专业评测集。")
    A("- **上线前必须补真实语料**：按 `../01_分布分析/意图节点命中明细.csv` 中"
      "「是否命中=否」的 445 个节点补标真实用户语句（每节点 ≥5 条、含多轮上下文），"
      "或用教师模型生成后人工校验，再与本数据集合并训练。")
    A("- export 中另外 %d 条 `是否无效意图=是` 且标注为空的 user 行是真实客户语句但无标签，"
      "可用于**继续预训练 / 领域适配**，不要当 SFT 监督信号。" %
      info["unlabeled_invalid_user_rows"])
    A("")
    A("**三条可执行的路径（按推荐度）**：")
    A("")
    A("| 路径 | 做法 | 代价 |")
    A("| :--- | :--- | :--- |")
    A("| A · 补标后训练（推荐） | 按 `../01_分布分析/意图节点命中明细.csv` 中零命中的 445 个节点补标真实"
      "用户语句（每节点 ≥5 条，含多轮上下文），与本数据集合并训练 | 需要标注人力，但结果可信 |")
    A("| B · 冷启动先行 | 先用本数据集微调，让 Qwen3-8B 学会输出格式与 495 个标签，"
      "上线前再用真实日志做一轮增量微调 | 快，但中间态不可上线 |")
    A("| C · 用 `--include-eval` | 用 432 条真实标注训练，同时另建独立测试集 | "
      "**原评测集作废**，需重建评测链路 |")
    A("")
    A("无论选哪条，最终效果都必须用一套**训练期间完全未见过的会话**来评，"
      "命令见 `../03_Qwen3-8B评测/Qwen3-8B专业能力评测说明.md`。")
    A("")
    A("## 七、文件与复现")
    A("")
    A("| 文件 | 说明 |")
    A("| :--- | :--- |")
    A("| `sft_train.jsonl` / `sft_dev.jsonl` / `sft_test.jsonl` | 训练 / 验证 / 内部测试 |")
    A("| `sft_labels.json` | 498 路径 + 495 末级标签，训练前后一致性校验用 |")
    if info["include_eval_generated"]:
        A("| `sft_train_with_eval.jsonl` | ⚠️ 含 432 条冻结评测样本的变体，选用前必须另建测试集（见 4.1） |")
    A("| `sft_dataset_info.json` | 规模、来源分布、防泄漏校验、各文件 sha256 |")
    A("| `build_sft_dataset.py` | 可复现脚本 |")
    A("| `微调数据集说明.md` | 本文档 |")
    A("")
    A("```bash")
    A('cd "D:/工作/文件/模型评测"')
    A("python result/02_微调数据集/build_sft_dataset.py                     # 默认严格隔离 + full 候选树")
    A("python result/02_微调数据集/build_sft_dataset.py --tree-mode names   # 只给末级名称")
    A("python result/02_微调数据集/build_sft_dataset.py --tree-mode none    # 不给候选（需足量真实语料后才用）")
    A("python result/02_微调数据集/build_sft_dataset.py --include-eval      # 额外产出被污染的版本（谨慎）")
    A("```")
    A("")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description="构建 Qwen3-8B 语音导航意图识别 SFT 数据集")
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--per-node", type=int, default=2, help="每个意图节点生成的问法条数")
    ap.add_argument("--max-noise", type=int, default=60, help="真实 ASR 噪声样本上限")
    ap.add_argument("--tree-mode", choices=["full", "names", "none"], default="full")
    ap.add_argument("--include-eval", action="store_true",
                    help="额外产出含 432 条冻结评测样本的数据集（会污染测试集）")
    return build(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
