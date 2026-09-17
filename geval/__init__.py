"""geval —— 轻量级大模型通用能力评测框架（零第三方依赖）。

设计目标：
  1. 同一套题目、同一套 prompt、同一套抽取与打分，横向对比多个模型；
  2. 不只输出准确率，同时输出置信区间、显著性检验与时延分布，
     让「小模型 + 微调是否值得」成为一个可量化决策；
  3. 纯标准库实现，可在内网 ARM64 离线环境直接跑。

最小用法：
    python -m geval selftest              # 离线端到端自检（mock 模型）
    python -m geval list-tasks            # 查看内置任务
    python -m geval preview --task ceval  # 打印真实 prompt 样例
    python -m geval run --config config.json
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
