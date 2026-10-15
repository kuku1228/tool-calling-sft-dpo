"""agent-optimization 核心库。

模块地图
--------
======================  ==================================================
模块                     职责
======================  ==================================================
``src.config``           全局配置单例 CFG（路径 / 模型 / 采样 / 超参）
``src.env.tool_catalog`` 6 个航空客服工具：schema + 模拟实现 + 白名单
``src.data.task_manifest`` 100 个评测任务：类别 / 难度 / 可见性 / 纠错轮
``src.eval.grader``      把一条策略输出判成 SUCCESS 或某一类错误
``src.eval.equivalence`` 参数级语义等价判定（日期、城市、ID、数值）
``src.eval.error_taxonomy`` 错误分类体系与分布统计
``src.eval.metrics``     pass@k / pass^k / bootstrap CI / 统计量
``src.eval.pass_k_eval`` 把采样结果聚合成完整评测报告
======================  ==================================================

快速开始
--------
>>> from src.config import CFG, describe
>>> len(describe()) > 0          # describe() 打印环境摘要，这里只验证它能跑
True
>>> from src.data.task_manifest import load_tasks
>>> tasks = load_tasks()
>>> len(tasks)
100
>>> sorted({t.difficulty for t in tasks})
['easy', 'hard', 'medium']
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
