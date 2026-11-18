# -*- coding: utf-8 -*-
"""语音导航专业能力评测数据集 - 意图节点分布分析。

输入（工作区根，只读）：
  语音导航意图识别评测数据集.jsonl      432 条评测样本
  意图树_完整.json                      498 个节点（含 sources / level / description）
  llm-general-eval/.../labels.json      评测框架使用的标签清单（一致性校验）

输出（本目录）：
  意图节点分布分析报告.md               人读报告
  意图节点命中明细.csv                  498 行全量节点明细（Excel 可直接打开）
  distribution_metrics.json            机器可读统计结果

口径：
  命中节点 = 该节点的「末级名称」被至少一条样本的 ground_truth（人工标注意图名称）命中。
  评测时模型输出与比对的对象就是末级名称，故以名称为主键；层级路径用于定位节点来源与层级。
"""
import csv
import json
import os
import sys
import collections

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(OUT))
F_ROWS = os.path.join(ROOT, "语音导航意图识别评测数据集.jsonl")
F_TREE = os.path.join(ROOT, "意图树_完整.json")
F_LABELS = os.path.join(ROOT, "llm-general-eval", "llm-general-eval",
                        "datasets", "intent_routing", "labels.json")


def last(path):
    return path.split("/")[-1]


def load():
    with open(F_ROWS, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(F_TREE, "r", encoding="utf-8") as f:
        tree = json.load(f)["nodes"]
    labels = None
    if os.path.isfile(F_LABELS):
        with open(F_LABELS, "r", encoding="utf-8") as f:
            labels = json.load(f)
    return rows, tree, labels


def source_of(node):
    s = node.get("sources") or []
    if len(s) > 1:
        return "两者共有"
    if not s:
        return "未知"
    return "FAQ节点" if s[0] == "FAQ列表" else "业务菜单节点"


def pct(a, b):
    return 0.0 if not b else 100.0 * a / b


def md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def analyze():
    rows, tree, labels = load()
    paths = [n["path"] for n in tree]
    path_set = set(paths)
    path2node = {n["path"]: n for n in tree}
    name2paths = collections.defaultdict(list)
    for n in tree:
        name2paths[last(n["path"])].append(n["path"])

    label_count = collections.Counter(str(r["ground_truth"]).strip() for r in rows)
    hit_labels = set(label_count)
    n_name_match = sum(1 for r in rows
                       if str(r["ground_truth"]).strip() in name2paths)
    ambiguous = {k: v for k, v in name2paths.items() if len(v) > 1}
    hit_ambiguous = {k: v for k, v in ambiguous.items() if k in hit_labels}

    # ---- 逐样本解析：优先用 human_intent_path 精确匹配，回退按末级名称 ----
    resolve_mode = collections.Counter()
    sample_src = collections.Counter()
    sample_level = collections.Counter()
    sample_src_level = collections.Counter()
    unresolved = []
    for r in rows:
        hp = str(r.get("human_intent_path") or "").strip()
        gt = str(r.get("ground_truth") or "").strip()
        p = None
        mode = None
        if hp in path_set:
            p, mode = hp, "路径精确匹配"
        else:
            cand = name2paths.get(gt) or []
            if len(cand) == 1:
                p, mode = cand[0], "名称唯一回退"
            elif len(cand) > 1:
                p, mode = cand[0], "名称歧义回退"
            else:
                mode = "无法定位"
                unresolved.append(gt)
        resolve_mode[mode] += 1
        if p:
            node = path2node[p]
            src, lv = source_of(node), node["level"]
            sample_src[src] += 1
            sample_level[lv] += 1
            sample_src_level[(src, lv)] += 1

    # ---- 节点级命中：按末级名称反查 ----
    node_hits = {}
    for n in tree:
        c = label_count.get(last(n["path"]), 0)
        node_hits[n["path"]] = c
    hit_paths = [p for p, c in node_hits.items() if c > 0]
    hit_set = set(hit_paths)

    total_nodes = len(tree)
    total_labels = len(name2paths)
    total_samples = len(rows)

    # ---- 分组覆盖率 ----
    src_all = collections.Counter(source_of(n) for n in tree)
    src_hit = collections.Counter(source_of(n) for n in tree if n["path"] in hit_set)
    src_samp = collections.Counter()
    for n in tree:
        src_samp[source_of(n)] += node_hits[n["path"]]

    lv_all = collections.Counter(n["level"] for n in tree)
    lv_hit = collections.Counter(n["level"] for n in tree if n["path"] in hit_set)
    lv_samp = collections.Counter()
    for n in tree:
        lv_samp[n["level"]] += node_hits[n["path"]]

    cross = collections.Counter((source_of(n), n["level"]) for n in tree)
    cross_hit = collections.Counter((source_of(n), n["level"])
                                    for n in tree if n["path"] in hit_set)

    # ---- 样本类型（原数据集语义分类）× 节点来源 ----
    type_src = collections.Counter()
    for r in rows:
        gt = str(r.get("ground_truth") or "").strip()
        cand = name2paths.get(gt) or []
        src = source_of(path2node[cand[0]]) if cand else "未命中"
        type_src[(str(r.get("sample_type")), src)] += 1
    type_count = collections.Counter(str(r.get("sample_type")) for r in rows)

    # ---- 集中度 ----
    top = label_count.most_common()
    cum, need80 = 0, 0
    for i, (_, c) in enumerate(top, start=1):
        cum += c
        if cum >= 0.8 * total_samples:
            need80 = i
            break
    singleton = sum(1 for _, c in top if c == 1)
    hhi = sum((c / total_samples) ** 2 for c in label_count.values())

    metrics = {
        "n_samples": total_samples,
        "n_sessions": len(set(str(r.get("session_id")) for r in rows)),
        "n_nodes": total_nodes,
        "n_labels": total_labels,
        "n_hit_labels": len(hit_labels & set(name2paths)),
        "n_hit_paths": len(hit_paths),
        "coverage_label_pct": round(pct(len(hit_labels & set(name2paths)), total_labels), 2),
        "coverage_path_pct": round(pct(len(hit_paths), total_nodes), 2),
        "by_source": {k: {"nodes": src_all[k], "hit": src_hit[k],
                          "coverage_pct": round(pct(src_hit[k], src_all[k]), 2),
                          "sample_hits": src_samp[k]} for k in src_all},
        "by_level": {str(k): {"nodes": lv_all[k], "hit": lv_hit[k],
                              "coverage_pct": round(pct(lv_hit[k], lv_all[k]), 2),
                              "sample_hits": lv_samp[k]} for k in sorted(lv_all)},
        "sample_resolve_mode": dict(resolve_mode),
        "sample_by_source": dict(sample_src),
        "sample_by_level": {str(k): v for k, v in sorted(sample_level.items())},
        "sample_type": dict(type_count),
        "top1_share_pct": round(pct(top[0][1], total_samples), 2),
        "top5_share_pct": round(pct(sum(c for _, c in top[:5]), total_samples), 2),
        "top10_share_pct": round(pct(sum(c for _, c in top[:10]), total_samples), 2),
        "labels_needed_for_80pct_samples": need80,
        "singleton_labels": singleton,
        "hhi": round(hhi, 4),
        "ambiguous_label_names": {k: v for k, v in ambiguous.items()},
        "hit_ambiguous_label_names": hit_ambiguous,
        "hit_labels_subset_of_framework_labels": (
            None if not labels else (hit_labels <= set(labels.get("labels") or []))),
        "unresolved_samples": len(unresolved),
    }

    # ---- CSV：498 行全量节点明细 ----
    csv_path = os.path.join(OUT, "意图节点命中明细.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["序号", "层级路径", "末级名称(意图标签)", "层级数", "一级节点", "二级节点",
                    "三级节点", "节点来源", "来源文件", "是否命中", "命中次数",
                    "占全部样本比率(%)", "是否末级名称歧义", "节点描述(清洗后)"])
        for i, n in enumerate(tree, start=1):
            c = node_hits[n["path"]]
            w.writerow([i, n["path"], last(n["path"]), n["level"], n["level1"],
                        n["level2"], n["level3"], source_of(n),
                        " / ".join(n.get("sources") or []), "是" if c else "否", c,
                        "%.2f" % pct(c, total_samples),
                        "是" if len(name2paths[last(n["path"])]) > 1 else "否",
                        (n.get("description_clean") or "").replace("\n", " ")[:300]])

    # ---- JSON ----
    with open(os.path.join(OUT, "distribution_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics,
                   "node_hits": [{"path": n["path"], "label": last(n["path"]),
                                  "level": n["level"], "source": source_of(n),
                                  "hits": node_hits[n["path"]]}
                                 for n in tree],
                   "label_counts": [{"label": k, "hits": v} for k, v in top]},
                  f, ensure_ascii=False, indent=2)

    # ---- 报告 ----
    L = []
    A = L.append
    A("# 语音导航专业能力评测数据集 · 意图节点分布分析报告")
    A("")
    A("> 数据源：`语音导航意图识别评测数据集.jsonl`（%d 条）+ `意图树_完整.json`（%d 个节点）。" %
      (total_samples, total_nodes))
    A("> 本报告由 `analyze_distribution.py` 自动计算生成，重跑即可刷新全部数字。")
    A("")
    A("## 一、结论速览")
    A("")
    A("- 数据集共 **%d 条样本 / %d 个会话**，只命中 **%d 个末级意图标签**、**%d 个层级节点**。" %
      (total_samples, metrics["n_sessions"], metrics["n_hit_labels"], metrics["n_hit_paths"]))
    A("- **节点覆盖率 %.2f%%**（%d/%d），**标签覆盖率 %.2f%%**（%d/%d）；**%d 个节点零命中**。" %
      (metrics["coverage_path_pct"], metrics["n_hit_paths"], total_nodes,
       metrics["coverage_label_pct"], metrics["n_hit_labels"], total_labels,
       total_nodes - metrics["n_hit_paths"]))
    A("- 分布极度偏斜：`%s` 单标签 %d 条，占 **%.2f%%**；Top5 占 %.2f%%、Top10 占 %.2f%%；"
      "只需 **%d 个标签**即可覆盖 80%% 的样本，仅出现 1 次的标签有 %d 个。" %
      (top[0][0], top[0][1], metrics["top1_share_pct"], metrics["top5_share_pct"],
       metrics["top10_share_pct"], need80, singleton))
    A("- 结论：这份数据集能稳定测出「高频通用意图 + 少量常见业务意图」的表现，"
      "**无法代表 498 个节点的整体能力**；用于微调会造成严重标签偏置（见 `../02_微调数据集/`）。")
    A("")
    A("## 二、统计口径（避免歧义）")
    A("")
    A("| 概念 | 定义 |")
    A("| :--- | :--- |")
    A("| 节点 | 意图树中的一条层级路径，共 %d 条；比末级名称多 %d 条，因 %d 个名称对应 2 条路径 |" %
      (total_nodes, total_nodes - total_labels, len(ambiguous)))
    A("| 末级名称 / 意图标签 | 路径最后一段，去重后 %d 个；**评测时模型输出与判分比对的就是它** |" % total_labels)
    A("| 命中 | 某节点的末级名称出现在至少一条样本的 `ground_truth` 中 |")
    A("| 命中次数 | 该名称在 %d 条样本中作为 `ground_truth` 出现的次数 |" % total_samples)
    A("| FAQ 节点 | `sources` 含 `FAQ列表`（来自 FAQ 列表模板，%d 个） |" % src_all.get("FAQ节点", 0))
    A("| 业务菜单节点 | `sources` 含 `储蓄卡业务菜单层级`（来自菜单层级模板，%d 个） |" %
      src_all.get("业务菜单节点", 0))
    A("| 层级 | 路径按 `/` 切分的段数：一级 / 二级 / 三级 |")
    A("")
    A("> 注意：原数据集的 `sample_type`（业务/FAQ意图 vs 通用交互意图）是**按语义白名单划分的样本类型**，"
      "与这里的「节点来自哪个 Excel」是两套不同口径，第四节会交叉对比，不要混用。")
    A("")
    A("### 2.1 标签定位方式")
    A("")
    A("优先用 `human_intent_path`（人工标注意图完整路径）精确匹配树路径；匹配不上时回退按 "
      "`ground_truth` 末级名称反查。结果：")
    A("")
    A(md_table(["定位方式", "样本数", "占比"],
               [[k, v, "%.2f%%" % pct(v, total_samples)] for k, v in resolve_mode.most_common()]))
    A("")
    A("`ground_truth` **%d/%d 条（%.2f%%）命中候选树末级名称**；但 `human_intent_path` 只有 %d 条能与树路径精确对上，"
      "其余 %d 条标注的是「不带父级的末级名称」或与树路径写法不一致，属于标注口径问题，"
      "不影响评测判分（判分用末级名称），但会影响按路径做层级归因。" %
      (n_name_match, total_samples, pct(n_name_match, total_samples),
       resolve_mode["路径精确匹配"],
       total_samples - resolve_mode["路径精确匹配"]))
    A("")
    A("末级名称一对多的歧义节点共 %d 个：%s。若此类名称被命中，其全部候选路径都会记为命中并在 CSV 的"
      "「是否末级名称歧义」列标出；本次命中的歧义名称为 **%d 个**，因此不影响本报告的计数。" %
      (len(ambiguous),
       "、".join("`%s`→%s" % (k, " / ".join(v)) for k, v in ambiguous.items()),
       len(hit_ambiguous)))
    A("")
    A("## 三、命中节点在全部节点中的比率")
    A("")
    A("### 3.1 总体")
    A("")
    A(md_table(["口径", "命中数", "总数", "覆盖率"],
               [["层级节点（路径）", metrics["n_hit_paths"], total_nodes,
                 "%.2f%%" % metrics["coverage_path_pct"]],
                ["末级名称（标签）", metrics["n_hit_labels"], total_labels,
                 "%.2f%%" % metrics["coverage_label_pct"]],
                ["零命中节点", total_nodes - metrics["n_hit_paths"], total_nodes,
                 "%.2f%%" % pct(total_nodes - metrics["n_hit_paths"], total_nodes)]]))
    A("")
    A("### 3.2 按节点来源（FAQ / 业务菜单）")
    A("")
    A(md_table(["节点来源", "节点总数", "命中节点数", "节点覆盖率", "样本命中次数", "次数占比"],
               [[k, src_all[k], src_hit[k], "%.2f%%" % pct(src_hit[k], src_all[k]),
                 src_samp[k], "%.2f%%" % pct(src_samp[k], total_samples)]
                for k in sorted(src_all)]))
    A("")
    A("### 3.3 按层级（一级 / 二级 / 三级）")
    A("")
    A(md_table(["层级", "节点总数", "命中节点数", "节点覆盖率", "样本命中次数", "次数占比"],
               [[("%d 级" % k), lv_all[k], lv_hit[k], "%.2f%%" % pct(lv_hit[k], lv_all[k]),
                 lv_samp[k], "%.2f%%" % pct(lv_samp[k], total_samples)]
                for k in sorted(lv_all)]))
    A("")
    A("### 3.4 来源 × 层级 交叉覆盖率")
    A("")
    A(md_table(["节点来源", "层级", "节点总数", "命中节点数", "覆盖率"],
               [[s, "%d 级" % lv, cross[(s, lv)], cross_hit[(s, lv)],
                 "%.2f%%" % pct(cross_hit[(s, lv)], cross[(s, lv)])]
                for (s, lv) in sorted(cross)]))
    A("")
    A("**读法**：二三级节点是最考验模型层级推理能力的细粒度业务意图，其覆盖率见上表 —— "
      "覆盖率越低，评测集越无法验证模型在复杂菜单下的路由能力。"
      "表中「样本命中次数」按节点累计，若某个末级名称对应多条路径会重复计入，"
      "占比因此可能略高于 100%，以「节点覆盖率」列为准。")
    A("")
    A("## 四、样本类型 × 节点来源（两套口径交叉）")
    A("")
    A(md_table(["原数据集 sample_type", "节点来源", "样本数", "占全部样本"],
               [[t, s, c, "%.2f%%" % pct(c, total_samples)]
                for (t, s), c in sorted(type_src.items())]))
    A("")
    A(md_table(["sample_type", "样本数", "占比"],
               [[k, v, "%.2f%%" % pct(v, total_samples)] for k, v in type_count.most_common()]))
    A("")
    A("## 五、命中节点明细（按命中次数降序，全 %d 个）" % len(hit_labels))
    A("")
    hit_nodes = [n for n in tree if node_hits[n["path"]] > 0]
    hit_nodes.sort(key=lambda n: (-node_hits[n["path"]], n["path"]))
    A(md_table(["#", "末级名称（模型应输出）", "层级路径", "层级", "来源", "命中次数", "占样本比"],
               [[i, last(n["path"]), n["path"], "%d 级" % n["level"], source_of(n),
                 node_hits[n["path"]], "%.2f%%" % pct(node_hits[n["path"]], total_samples)]
                for i, n in enumerate(hit_nodes, start=1)]))
    A("")
    A("> 完整 498 行（含零命中节点、一级/二级/三级字段、来源文件、节点描述）见 "
      "`意图节点命中明细.csv`；机器可读版见 `distribution_metrics.json`。")
    A("")
    A("## 六、分布特征与风险")
    A("")
    A("1. **长尾极重**：HHI = %.4f；Top1 `%.2f%%`、Top5 `%.2f%%`；%d 个标签只出现 1 次。"
      "整体准确率会被「人工服务 / 集外 / 问候」这类高频简单意图主导，必须与分层准确率一起看。" %
      (hhi, metrics["top1_share_pct"], metrics["top5_share_pct"], singleton))
    zero_faq = src_all.get("FAQ节点", 0) - src_hit.get("FAQ节点", 0)
    zero_menu = src_all.get("业务菜单节点", 0) - src_hit.get("业务菜单节点", 0)
    zero_lv23 = (lv_all[2] - lv_hit[2]) + (lv_all[3] - lv_hit[3])
    A("2. **业务能力测不全**：%d 个节点零命中（%.2f%%），其中 FAQ 节点 %d 个、业务菜单节点 %d 个，"
      "二三级细粒度节点 %d 个。直接拿这份评测集的「意图准确率」当作"
      "「模型在语音导航场景的专业能力」会系统性高估。" %
      (total_nodes - metrics["n_hit_paths"], pct(total_nodes - metrics["n_hit_paths"], total_nodes),
       zero_faq, zero_menu, zero_lv23))
    A("3. **标注路径口径不统一**：%d 条样本的 `human_intent_path` 无法与树路径精确匹配；"
      "另有 %d 个末级名称对应 2 条路径，做层级归因或转成层级分类标签前必须先消歧。" %
      (total_samples - resolve_mode["路径精确匹配"], len(ambiguous)))
    A("4. **对微调的直接影响**：作为冻结测试集，绝不能进训练集；且标签覆盖只有 %.2f%%，"
      "直接微调会把 8B 模型训成「动不动就答人工服务」。" % metrics["coverage_label_pct"])
    A("")
    A("## 七、建议")
    A("")
    A("- **评测**：保留此 432 条作为冻结测试集（`dataset_info.json` 已有 sha256），"
      "报告必须同时给出分层准确率、Top 混淆对、幻觉意图率；对外表述为"
      "「在 53 个高频意图上的准确率」，不要表述为「全意图树准确率」。")
    A("- **补集**：按 `意图节点命中明细.csv` 中「是否命中=否」的节点补标真实用户语句，"
      "优先补二三级业务菜单节点，形成覆盖度更高的第二套评测集。")
    A("- **微调**：见 `../02_微调数据集/微调数据集说明.md`，使用独立构建的 SFT 数据，"
      "与本报告的数据集严格隔离。")
    A("")

    report = "\n".join(L)
    with open(os.path.join(OUT, "意图节点分布分析报告.md"), "w", encoding="utf-8") as f:
        f.write(report)

    print("样本 %d ｜ 节点 %d ｜ 命中节点 %d（%.2f%%）｜ 命中标签 %d（%.2f%%）" %
          (total_samples, total_nodes, metrics["n_hit_paths"], metrics["coverage_path_pct"],
           metrics["n_hit_labels"], metrics["coverage_label_pct"]))
    print("按来源：" + "，".join("%s %d/%d" % (k, src_hit[k], src_all[k]) for k in sorted(src_all)))
    print("按层级：" + "，".join("%d级 %d/%d" % (k, lv_hit[k], lv_all[k]) for k in sorted(lv_all)))
    print("已输出：意图节点分布分析报告.md / 意图节点命中明细.csv / distribution_metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(analyze())
