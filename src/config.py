"""全局配置：路径、模型、采样参数、日志埋点全部集中在这里。

设计原则
--------
1. **单一事实来源**：任何脚本都不应该硬编码路径或超参会出现在别处的值，
   一律 `from src.config import CFG`。
2. **零重依赖**：本模块只依赖标准库。YAML 解析优先用 PyYAML，缺失时
   降级到 `json.loads`（本项目的 yaml 配置文件都写成 JSON 兼容子集）。
   这样 `import src.config` 在裸 CPU 环境永远成功。
3. **可覆盖**：所有字段都能用环境变量覆盖（见 `_env_override`），
   方便在 AutoDL 上不改进仓库代码就切模型路径。

目录约定
--------
::

    agent-optimization/
    ├── configs/              # yaml 配置（每行一条超参）
    ├── data/                 # 生成的偏好数据 / SFT 数据（不入库）
    ├── docs/                 # 报告与文档
    ├── experiments/          # 评测任务定义 + 评测结果
    ├── outputs/              # 训练输出（checkpoint、metrics、曲线）
    ├── scripts/              # 入口脚本
    ├── src/                  # 核心库
    └── tests/                # 单元测试

常用命令
--------
>>> from src.config import CFG
>>> CFG.model.name
'Qwen/Qwen2.5-1.5B-Instruct'
>>> CFG.path("experiments", "baseline_airline").name
'baseline_airline'
>>> CFG.path("experiments", "baseline_airline").parent.name
'experiments'
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# ============================================================================
# 1. 路径解析
# ============================================================================

#: 仓库根目录 = 本文件的上上级（src/config.py -> src -> 仓库根）
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: 是否运行在 Windows 上。远程 GPU 机器（AutoDL）是 Linux，
#: 少数地方需要区分（例如 shell 脚本的执行方式）。
IS_WINDOWS: bool = sys.platform.startswith("win")

#: 是否强制 mock 模式。设置 `AGENTOPT_FORCE_MOCK=1` 后，
#: 所有会加载真实模型的代码路径都会退化为 mock，便于在 CPU 上验证流水线。
FORCE_MOCK: bool = os.environ.get("AGENTOPT_FORCE_MOCK", "0") not in ("0", "", "false", "False")


# ============================================================================
# 2. 配置数据类
# ============================================================================


@dataclass
class PathsCfg:
    """所有相对路径都相对仓库根解析。"""

    data: str = "data"
    docs: str = "docs"
    experiments: str = "experiments"
    outputs: str = "outputs"
    configs: str = "configs"
    logs: str = "logs"


@dataclass
class ModelCfg:
    """模型与推理设置。

    小模型三个档位，按显存选：

    ==========================  ==========  ========================
    模型                         参数量      适用场景
    ==========================  ==========  ========================
    Qwen2.5-0.5B-Instruct       0.5B        CPU debug / 显存 < 8G
    Qwen2.5-1.5B-Instruct       1.5B        单卡 3090 24G（本项目主线）
    Qwen2.5-3B-Instruct         3B          单卡 4090 24G（备选）
    ==========================  ==========  ========================
    """

    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    #: 评测时采样温度。0.7 是 τ-bench 的设定，保留随机性以体现 pass^k 的稳定性差异。
    temperature: float = 0.7
    top_p: float = 0.9
    #: 单轮最大生成 token。1024 对小模型的「长思考 + tool_call」经常截断，
    #: 这里给到 1536，仍远小于训练的 max_length。
    max_new_tokens: int = 1536
    #: 训练/评测统一的最大序列长度（prompt + completion）。
    max_seq_len: int = 2048
    #: 推理精度。bf16 在 30/40 系都支持；CPU 上忽略。
    dtype: str = "bfloat16"
    device: str = "auto"  # auto | cuda | cpu | mps
    #: 本地权重路径覆盖。若设置了非空值，优先于 name。
    #: 用途：AutoDL 上先把模型下到数据盘，避免每个 step 都走 HF 网络。
    local_path: str = ""
    #: HF 镜像端点（国内机器常用 https://hf-mirror.com）
    hf_endpoint: str = ""

    @property
    def short_name(self) -> str:
        """用于文件名与日志的短名，如 ``qwen2.5-1.5b-instruct``。"""
        return self.name.split("/")[-1].lower().replace("_", "-")

    @property
    def resolved(self) -> str:
        """最终加载路径：本地路径优先。"""
        return self.local_path or self.name


@dataclass
class EvalCfg:
    """评测协议。改这里的值会让所有历史实验失去可比性，改动需谨慎。"""

    #: 每个任务采样次数。pass^k 需要 k <= num_samples。
    num_samples_per_task: int = 4
    #: 报告哪些 k。1 用于主指标对齐，4 用于稳定性。
    pass_k_values: tuple[int, ...] = (1, 2, 4)
    #: 主指标用哪个 k 的 pass^k。
    primary_k: int = 4
    #: 自建评测集的任务数（生成后的总数）。
    num_tasks: int = 100

    def __post_init__(self) -> None:
        if self.primary_k > self.num_samples_per_task:
            raise ValueError(
                f"primary_k={self.primary_k} 超过 num_samples_per_task={self.num_samples_per_task}，"
                "pass^k 无法估计。请提高采样次数或降低 primary_k。"
            )


@dataclass
class TrainCfg:
    """DPO 训练超参默认值。yaml 里只写与默认值不同的项。"""

    # --- DPO 核心 ---
    #: β 控制对参考模型的偏离程度。β 越大越保守，越小越大胆。
    #: 1.5B 小模型上 0.1 通常比 0.5 更有效（模型本身能力弱，需要更大的更新幅度）。
    beta: float = 0.1
    loss_type: str = "sigmoid"  # sigmoid | ipo | hinge | kto_pair ...
    label_smoothing: float = 0.0
    #: DPO 论文的 reference-model 自由形式（省一次前向）。
    reference_free: bool = False

    # --- 优化器 ---
    lr: float = 5e-5
    epochs: int = 2
    batch_size: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    #: "all-linear" 会覆盖 q/k/v/o + MLP，比手写列表省心。
    lora_target_modules: str = "all-linear"

    # --- 显存 ---
    gradient_checkpointing: bool = True
    bf16: bool = True
    #: DPO 的 logprob 计算峰值显存约是普通 LM 的 2-3 倍，序列长度要收着点。
    #:
    #: ⚠️ **这两个默认值必须能容下本项目的真实数据，不能只是「看起来合理」。**
    #:
    #: 实际遇到过的问题：默认值原本是 ``max_length=1536 / max_prompt_length=1024``，
    #: 而 ``dpo_train.py`` 在没传 ``--config`` 时会静默回退到这两个值。
    #: 实测后果：**23.2% 的 prompt 被截断**（实测 prompt 全量 max=1136 > 1024），
    #: system prompt 里的工具定义被切掉，模型学不到完整参数说明 ——
    #: 而 loss 曲线、截断报告（当时只报 completion）全都显示正常。
    #:
    #: **默认值不能落在「会静默损坏数据」的区间。**
    #: 现在按实测长度设定：prompt p99=1097 / max=1136，
    #: prompt+completion max=1137。取 1280 / 1152 留出安全余量。
    #: ``tests/test_core.py`` 有断言守住这个下限。
    max_length: int = 1280
    max_prompt_length: int = 1152

    # --- 日志与保存 ---
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 3
    seed: int = 42
    report_to: str = "tensorboard"


@dataclass
class PreferenceCfg:
    """偏好数据构造参数。"""

    #: 目标条数（按类型）
    n_format: int = 1200
    n_recovery: int = 1200
    n_tool_selection: int = 600
    #: 「该追问」边界对的条数。
    #:
    #: ⚠️ 这一类的配比是**真实基线数据定的**，不是拍脑袋：
    #: 实测 SFT 之后剩下的唯一失败类型就是「信息不足时应先追问，却直接
    #: 调用了工具」（20 条，占失败 100%）。所以它虽然条数少，
    #: 却是 DPO 唯一有明确改善目标的类别。
    #:
    #: 条数受任务集限制：``expected.tool is None`` 的任务只有 5 个，
    #: 因此实际能构造出的条数远小于这个目标值（其余用合成场景兜底）。
    n_clarify: int = 400
    #: 全局随机种子，保证数据可复现
    seed: int = 42
    #: 同一 prompt 最多生成多少条偏好对
    max_pairs_per_task: int = 8
    #: 是否对 rejected 做「难度分层」：保留一部分「看起来合理但实际错」的 hard negative
    hard_negative_ratio: float = 0.35


@dataclass
class MockPolicyCfg:
    """Mock 策略的误差率参数。

    ⚠️ **重要说明**：这些数字是**待验证的假设值**，不是本项目声称的结论。
    它们的唯一用途是在没有 GPU 的机器上把「评测 → 分类 → 偏好构造 → 训练 →
    再评测 → 画图」这条流水线跑通。真实数字一律来自
    ``experiments/*/eval_report.json``，并由
    ``scripts/eval/collect_results.py`` 汇总。

    概率模型（见 ``src/mock/mock_policy.py``）::

        p_success                   = 1 − Σ p_err(e)          ← 成功是补集
        p_err(e | 实验 X)           = baseline_error_rates[e]
                                       × mitigation_factors[X][e]
        p_err 归一化                 保证 Σ p_err ≤ 1

    「成功 = 1 − 各错误概率之和」这个结构很关键：
    只有让改善某一类错误**释放**出来的概率质量流向成功，
    各实验的准确率 / pass^k 才会真的产生差异。
    如果成功概率独立设定，那么「把格式错误降低 65%」只是
    把错误挪到别的类别，指标纹丝不动 —— 那就不是在建模改进，
    而是在自欺欺人。

    各误差率的物理含义（首轮、未训练）：
      - 模型对复杂格式指令的遵循能力弱 → 格式错误占比最高
      - 参数语义（中文城市名 vs 三字码、相对日期）容易错 → 参数错误次之
      - 工具列表给得清楚时幻觉不算最高频 → 中等
      - 6 个工具语义区分度尚可 → 工具选错最低
    """

    #: 未训练时的各类错误概率（首轮），总和应 < 1。
    #: key 一律用 ``src/eval/error_taxonomy.py`` 里的类别常量，
    #: 避免「配置里叫 format_error、代码里查 FORMAT_ERROR」这类静默失配。
    baseline_error_rates: dict[str, float] = field(
        default_factory=lambda: {
            "FORMAT_ERROR": 0.180,
            "HALLUCINATION": 0.105,
            "WRONG_TOOL": 0.075,
            "PARAM_ERROR": 0.160,
            "NO_RECOVERY": 0.060,
        }
    )

    #: 首轮就答对的额外加成，反映「任务越简单越可能一次就对」
    easy_task_bonus: float = 0.15
    hard_task_penalty: float = 0.08

    #: 各实验对各误差类型的乘性缩减系数（1.0 = 无改善）
    #: key = 实验名（baseline / sft_only / dpo_format / dpo_recovery / dpo_joint）
    #: 对齐 docs/ablation/ablation_plan.md §1.3 的实验矩阵
    mitigation_factors: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "baseline": {
                "FORMAT_ERROR": 1.00,
                "HALLUCINATION": 1.00,
                "WRONG_TOOL": 1.00,
                "PARAM_ERROR": 1.00,
                "NO_RECOVERY": 1.00,
            },
            # Exp 1：只训格式约束对 → 格式错误大降，工具名类错误基本不动
            "dpo_format": {
                "FORMAT_ERROR": 0.35,
                "HALLUCINATION": 0.95,
                "WRONG_TOOL": 0.92,
                "PARAM_ERROR": 0.90,
                "NO_RECOVERY": 0.95,
            },
            # Exp 2：只训错误恢复对 → 恢复能力提升，首轮错误降幅有限
            "dpo_recovery": {
                "FORMAT_ERROR": 0.95,
                "HALLUCINATION": 1.00,
                "WRONG_TOOL": 0.95,
                "PARAM_ERROR": 0.92,
                "NO_RECOVERY": 0.40,
            },
            # Exp 3：联合方案 → 两类效应叠加
            "dpo_joint": {
                "FORMAT_ERROR": 0.35,
                "HALLUCINATION": 0.40,
                "WRONG_TOOL": 0.75,
                "PARAM_ERROR": 0.70,
                "NO_RECOVERY": 0.35,
            },
            # 对照：SFT 只用 chosen 做监督（验证「对比信号」而不是
            # 「更多同分布监督」才是提升来源）。
            #
            # ⚠️ 这组系数是本项目**假设的、且可能被真实实验推翻的**结论：
            #   - SFT 能从 chosen 里学会格式与工具名（甚至学得不错），
            #     所以 FORMAT_ERROR / HALLUCINATION 的降幅与 dpo_format 相当；
            #   - 但 SFT **学不到「被纠正后改对」**，因为 chosen 里根本没有
            #     「先错后改」的样本 —— 所以 NO_RECOVERY 几乎不降（0.90）。
            # 这个差异正是本项目核心论点的可执行表达。真实实验如果发现
            # SFT 在恢复率上也大幅提升，则本项目论点不成立，结论需相应改写。
            "sft_only": {
                "FORMAT_ERROR": 0.40,
                "HALLUCINATION": 0.60,
                "WRONG_TOOL": 0.85,
                "PARAM_ERROR": 0.80,
                "NO_RECOVERY": 0.90,
            },
        }
    )

    #: DPO 训练对「恢复」的额外增益上限（在 mitigation 之上再乘一次）
    recovery_gain: dict[str, float] = field(
        default_factory=lambda: {
            "baseline": 1.0,
            "dpo_format": 0.95,
            "dpo_recovery": 2.30,
            "dpo_joint": 3.10,
            "sft_only": 1.15,
        }
    )

    #: 幻觉发生的条件下，编造的工具名从这个池子里采样
    hallucinated_tool_pool: tuple[str, ...] = (
        "search_hotel",
        "book_hotel",
        "get_flight_status",
        "refund_flight",
        "check_in_online",
        "get_loyalty_points",
        "transfer_to_human_agent",
        "send_confirmation_email",
    )


@dataclass
class Config:
    """顶层配置聚合。"""

    paths: PathsCfg = field(default_factory=PathsCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    preference: PreferenceCfg = field(default_factory=PreferenceCfg)
    mock: MockPolicyCfg = field(default_factory=MockPolicyCfg)

    # 全局
    seed: int = 42
    #: 实验标识，会写进所有产物，便于回溯「这份结果是谁跑的」
    run_tag: str = ""

    # ------------------------------------------------------------------
    # 路径工具
    # ------------------------------------------------------------------
    def path(self, *parts: str | Path) -> Path:
        """拼出绝对路径（相对仓库根）。

        优先按 ``self.paths`` 里的字段做映射，映射不到就直接用原字符串。
        注意本函数**不创建目录** —— 需要目录存在请用 ``out_dir`` 或
        ``ensure_dirs``。

        >>> CFG.path("experiments", "x").name
        'x'
        >>> CFG.path("docs").name
        'docs'
        """
        if not parts:
            return PROJECT_ROOT
        head, *tail = parts
        base = PROJECT_ROOT / getattr(self.paths, str(head), str(head))
        return base.joinpath(*[str(t) for t in tail])

    def out_dir(self, *parts: str) -> Path:
        """训练输出目录，自动创建。"""
        p = self.path("outputs", *parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def ensure_dirs(self) -> list[Path]:
        """创建所有约定的目录，返回被创建的目录列表。"""
        created: list[Path] = []
        for rel in (
            "data/preferences",
            "data/sft",
            "data/raw",
            "experiments",
            "outputs",
            "logs",
            "docs/figures",
        ):
            p = PROJECT_ROOT / rel
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
                created.append(p)
        return created

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        def _dump(obj: Any) -> Any:
            if is_dataclass(obj) and not isinstance(obj, type):
                return {f.name: _dump(getattr(obj, f.name)) for f in fields(obj)}
            if isinstance(obj, dict):
                return {k: _dump(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_dump(v) for v in obj]
            if isinstance(obj, Path):
                return str(obj)
            return obj

        return _dump(self)

    def dump_json(self, path: str | Path) -> Path:
        """把完整配置写盘，作为实验元数据的一部分。"""
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    # ------------------------------------------------------------------
    # 环境变量覆盖
    # ------------------------------------------------------------------
    def apply_env_overrides(self) -> Config:
        """让 AutoDL 上不改进仓库就能切模型/端点。

        支持的环境变量::

            AGENTOPT_MODEL=Qwen/Qwen2.5-3B-Instruct
            AGENTOPT_MODEL_LOCAL=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
            AGENTOPT_HF_ENDPOINT=https://hf-mirror.com
            AGENTOPT_DEVICE=cuda
            AGENTOPT_RUN_TAG=autodl-3090-r1
            AGENTOPT_SEED=1234
        """
        mapping = {
            "AGENTOPT_MODEL": ("model", "name"),
            "AGENTOPT_MODEL_LOCAL": ("model", "local_path"),
            "AGENTOPT_HF_ENDPOINT": ("model", "hf_endpoint"),
            "AGENTOPT_DEVICE": ("model", "device"),
            "AGENTOPT_DTYPE": ("model", "dtype"),
            "AGENTOPT_RUN_TAG": ("", "run_tag"),
        }
        for env_key, (section, attr) in mapping.items():
            val = os.environ.get(env_key)
            if not val:
                continue
            target = self if not section else getattr(self, section)
            setattr(target, attr, val)

        seed_env = os.environ.get("AGENTOPT_SEED")
        if seed_env and seed_env.isdigit():
            self.seed = int(seed_env)

        if self.model.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.model.hf_endpoint)
        return self


# ============================================================================
# 3. 轻量 YAML 读取（PyYAML 可选）
# ============================================================================


def strip_bom(text: str) -> str:
    """剥掉字符串开头的 UTF-8 BOM（``\\ufeff``）。

    单独抽出来是为了**可测**：``load_config_file`` 会读写真实文件，
    在受限沙箱里测不了；而这个纯字符串函数在哪都能跑。

    >>> strip_bom('\\ufeff{"a": 1}')
    '{"a": 1}'
    >>> strip_bom('{"a": 1}')          # 没有 BOM 时原样返回
    '{"a": 1}'
    >>> strip_bom("")
    ''
    """
    return text[1:] if text.startswith("\ufeff") else text


def _coerce_numeric_strings(obj: Any) -> Any:
    """把「看起来是数字的字符串」还原成数字。

    ⚠️ **这个函数存在的唯一原因是一个真实的、极隐蔽的解析器差异。**

    项目的 ``configs/*.yaml`` 写成 JSON 兼容子集，但读取时**优先用 PyYAML**、
    只在缺 PyYAML 时退回 ``json.loads``。而这两个解析器对
    **科学计数法**的处理不同::

        "lr": 5e-05          # 同一个文件内容

        json.loads  → 5e-05    (float)   ← JSON 规范允许无小数点指数
        yaml.safe_load → '5e-05' (str)   ← YAML 1.1 要求指数前有小数点

    实测后果：``lr='5e-05'``（字符串）被传进 ``torch.optim.AdamW``，
    在 ``if not 0.0 <= lr:`` 处抛
    ``TypeError: '<=' not supported between instances of 'float' and 'str'``。

    最阴险的地方是**它只在装了 PyYAML 的机器上出现**：
    开发机（无 PyYAML）走 json 路径一切正常，云端（conda 环境自带 PyYAML）
    直接崩。这也是为什么本地测试发现不了。

    这里做保守归一：只转换**整串都是数字**的字符串，
    ``"sigmoid"`` / ``"cosine"`` / ``"all-linear"`` 这类合法字符串不受影响。

    >>> _coerce_numeric_strings({"lr": "5e-05", "loss_type": "sigmoid"})
    {'lr': 5e-05, 'loss_type': 'sigmoid'}
    >>> _coerce_numeric_strings({"a": [{"b": "1.5"}]})
    {'a': [{'b': 1.5}]}
    >>> _coerce_numeric_strings("0.0001")
    0.0001
    >>> _coerce_numeric_strings("5e-05") == 5e-05
    True
    """
    if isinstance(obj, dict):
        return {k: _coerce_numeric_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numeric_strings(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return obj
        try:
            # int 优先：避免把 "42" 变成 42.0
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return obj
    return obj


def load_config_file(path: str | Path) -> dict[str, Any]:
    """读取 configs/ 下的 yaml。

    PyYAML 存在时用 ``yaml.safe_load``；否则退回 ``json.loads``。
    项目的 ``configs/*.yaml`` 都写成 JSON 兼容子集（键加引号、不用锚点），
    因此这个降级路径是可靠的，不是权宜之计。

    ⚠️ **两条路径的结果会被归一**（``_coerce_numeric_strings``）：
    PyYAML 与 JSON 对科学计数法的解释不同（见该函数的说明），
    不归一会导致「本地正常、云端崩」这种最难查的类型错误。

    ⚠️ **BOM 处理**：用 ``utf-8-sig`` 解码而不是 ``utf-8``，
    并用 ``strip_bom`` 再兜一层。

    原因：Windows 上用 PowerShell 的 ``Set-Content -Encoding UTF8`` 保存文件
    会写入 UTF-8 BOM，而 ``json.loads`` 遇到 BOM 会直接报
    ``Unexpected UTF-8 BOM``。这个坑很隐蔽 —— 文件在编辑器里看起来完全正常，
    只有**没装 PyYAML 的机器**上（也就是本项目刻意支持的「干净环境」）
    才会炸。``utf-8-sig`` 会自动剥掉 BOM，对没有 BOM 的文件行为一致。
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    text = strip_bom(p.read_text(encoding="utf-8-sig"))
    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(text) or {}
    except ImportError:
        raw = json.loads(text)
    return _coerce_numeric_strings(raw)


# ============================================================================
# 4. 单例
# ============================================================================

#: 全局配置单例。`from src.config import CFG`
CFG = Config().apply_env_overrides()


def describe() -> str:
    """给 CLI 打印用的环境摘要。"""
    cuda = "n/a"
    try:
        import torch  # type: ignore

        cuda = f"torch {torch.__version__}, cuda={torch.cuda.is_available()}"
        if torch.cuda.is_available():
            cuda += f", device={torch.cuda.get_device_name(0)}"
    except ImportError:
        cuda = "torch 未安装（Mock 模式可用）"

    return "\n".join(
        [
            "=" * 68,
            "agent-optimization 运行环境",
            "=" * 68,
            f"  仓库根目录 : {PROJECT_ROOT}",
            f"  平台       : {sys.platform}  (python {sys.version.split()[0]})",
            f"  模型       : {CFG.model.resolved}",
            f"  设备       : {CFG.model.device}",
            f"  采样       : T={CFG.model.temperature} top_p={CFG.model.top_p} "
            f"max_new_tokens={CFG.model.max_new_tokens}",
            f"  评测       : {CFG.eval.num_tasks} 任务 × {CFG.eval.num_samples_per_task} 采样, "
            f"pass^{CFG.eval.primary_k} 为主指标",
            f"  深度学习栈 : {cuda}",
            f"  强制 Mock  : {FORCE_MOCK}",
            "=" * 68,
        ]
    )


if __name__ == "__main__":
    print(describe())
