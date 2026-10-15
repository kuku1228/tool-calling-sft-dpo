# Prompt 契约：训练与推理必须同源

> **文档定位**：本文件定义「模型看到什么」这件事的唯一约定。
> 回答的问题：prompt 长什么样、由谁生成、训练与推理如何保证一致、
> 不一致会有什么后果、怎么自动检查。
>
> **读者**：要改 prompt、要加新工具、要排查「训练 loss 下降但评测指标不动」的人。
>
> **对应代码**：[`src/models/prompt_builder.py`](../../src/models/prompt_builder.py)（唯一定义处）

---

## §1 为什么需要一份「契约」

偏好对齐最常见的失效原因**不是超参不对，而是训练分布与推理分布不一致**。

具体到这里：如果训练时模型看到的是 A 版 system prompt、评测时看到的是 B 版，那么 DPO 学到的「在 A 版下要输出 `<tool_call>`」这个条件反射，在 B 版下不一定被激活。表现是：

```
训练 loss:  0.6931 → 0.42  → 0.31  → 0.28     ✅ 看起来一切正常
评测指标:   51.2%  → 51.0% → 51.3% → 51.1%    ❌ 完全不动
```

这种情况极难排查，因为**训练侧的所有信号都是健康的**。等到评测才发现，一轮 GPU 预算已经花掉了。

所以本项目把「prompt 从哪来」变成一条硬约束：

> **`src/models/prompt_builder.py` 是 prompt 的唯一事实来源。
> 任何其他模块都不得自己定义 system prompt 或模板。**

---

## §2 Prompt 的组成

一次推理的输入由三部分组成：

```
┌─────────────────────────────────────────────────────────────┐
│ 1. system prompt（含工具清单 + 格式规范 + 5 条行为规则）        │
├─────────────────────────────────────────────────────────────┤
│ 2. 对话历史（多轮，observation 以 user 角色注入）              │
├─────────────────────────────────────────────────────────────┤
│ 3. assistant 起始标记（模型从这里开始生成）                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 System prompt

定义在 `src/models/prompt_builder.py` 的 `SYSTEM_PROMPT`，通过 `build_system_prompt()` 渲染。结构：

1. **角色与工具清单**：由 `src/env/tool_catalog.py` 的 `render_tool_manual()` 生成。
   用人类可读的分行列表，而不是 JSON schema —— 1.5B 模型对「嵌套 JSON 的 schema」
   理解明显弱于对「分行列表」的理解。
2. **格式规范**：`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`，
   并明确「一次只输出一个工具调用」。
3. **5 条行为规则**：只能用列出的工具、参数不能填占位符、日期必须绝对格式、
   信息不足先追问、不需要工具时直接自然语言回复。

代码里有一处**有意的重复**：系统提示词同时以两种形式提供给模型 ——

- **JSON schema** 走 `apply_chat_template(tools=[...])` 的 `tools=` 通道
- **文本说明** 走 system prompt

但 `render_chat_text()` 的 `use_tools_channel` 参数**默认建议 False**。原因见 §3。

### 2.2 对话历史与 observation

任务由 `src/data/task_manifest.py` 的 `Task` 描述，`Task.to_messages()` 渲染历史。

**observation 以 `user` 角色注入，而不是单独的 `tool` 角色**：

```
<observation>{"ok": true, "result": {...}}</observation>
```

理由：1.5B 模型 + 简单 chat template 下，`tool` 角色的训练/推理一致性更难保证
（不同 tokenizer 的模板对 `tool` 角色处理差异较大）。把它作为 user 消息注入是
Hermes / Qwen 风格 tool calling 的常见做法，行为更可预测。

observation 保持 **JSON 而不是自然语言**：小模型对 JSON observation 的字段引用
更准确（多轮任务里要复用 `booking_id`、`flight_id`）。

### 2.3 为什么工具说明不通过 `tools=` 通道传

`render_chat_text()` 提供 `use_tools_channel` 开关，但**默认关闭**。

原因：Qwen2.5 的 chat template 对 `tools=` 有自己的一套渲染格式（模板自带的
function-calling 标记），而本项目训练的是 `<tool_call>` 格式。两者**不保证一致**。

一旦不一致，就回到了 §1 那个陷阱：训练与推理走了不同的渲染路径。

所以本项目的选择是：**把工具说明写在 system prompt 里，训练与推理走完全相同的文本路径**。
`tools=` 通道保留为可选项，供想对齐 Qwen 原生格式的后续实验使用。

---

## §3 唯一事实来源的实现方式

`src/data/preference_builder.py`（构造训练数据）**不定义自己的 system prompt**，
而是直接 import：

```python
# src/data/preference_builder.py
from src.models.prompt_builder import (
    SYSTEM_PROMPT,
    build_system_prompt,
    build_plain_prompt,
)

#: 向后兼容别名（早期版本叫 SYSTEM_PROMPT_TEMPLATE）
SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT
```

这样「训练数据的 prompt」与「评测时的 prompt」在编译期就是同一个对象。

### 3.1 自动一致性检查

光靠「共用同一个函数」还不够 —— 模板可能被间接改动（例如 `render_tool_manual`
的输出变了，或者任务集变了）。所以训练前会做一次**逐字比对**：

```
src/training/dpo_dataset.py :: verify_prompt_consistency()
```

做法：对每个带 `task_id` 且该 id 属于评测任务集的偏好样本，
用 `build_plain_prompt(task)` 重新渲染一次，比对前 400 字符。

用「前缀」而非全等比对的原因：恢复类偏好样本的 prompt 比评测 prompt
多出若干轮（首轮错误 + 纠错话术），但 **system prompt 与最初几轮对话必须完全一致**。

这个检查在 `scripts/train/dpo_train.py` 的数据体检阶段自动运行，不一致会**明确报错**：

```
✗ prompt 不一致 37 条 —— 训练与推理分布不同！
```

### 3.2 数据体检还会报什么

同一阶段（`data_health_check()`）还会检查：

| 检查项 | 不通过时的后果 |
|---|---|
| 数据质量审计（两侧语义等价、chosen 非 canonical 等） | 噪声样本，DPO loss 恒为 log 2 |
| token 长度分位数 | 用于确定 `max_length`，避免 OOM 或截断 |
| **截断后退化率** | 截断后两侧变相同的样本对 DPO 是纯噪声 |
| prompt 一致性 | 见 §3.1 |

---

## §4 改动 prompt 的操作规程

任何对 prompt 的改动都会让**此前所有实验结果失去可比性**。所以：

1. **先改代码**：只改 `src/models/prompt_builder.py`。
   不要在别处新增模板，不要在 yaml 里放 prompt 文本。
2. **跑一致性检查**：`python scripts/prepare/inspect_preferences.py`
   必须通过 prompt 一致性校验。
3. **重新构造偏好数据**：`python scripts/prepare/build_preferences.py`
   —— 旧数据是用旧 prompt 生成的，不能复用。
4. **重新跑基线与全部实验**：prompt 变了，基线也变了，旧对比表作废。
5. **记录改动**：在 `docs/ablation/ablation_report.md` 的「未达预期之处 / 变更记录」
   里写清改了什么、为什么改、旧结果是否作废。

> 想验证「prompt 详细程度」本身的影响（而不是 DPO 的影响）时，
> **不要**去改 `SYSTEM_PROMPT`，而应该用现成的消融开关：
>
> ```bash
> python scripts/eval/run_baseline.py --label baseline_lite --lite-system
> ```
>
> 它用的是 `SYSTEM_PROMPT_LITE`（精简版），与主 prompt 并列存在、
> 各自独立，不会污染主路径。

---

## §5 相关文件索引

| 关注点 | 文件 |
|---|---|
| system prompt 唯一定义 | `src/models/prompt_builder.py` |
| 工具 schema 与文本说明渲染 | `src/env/tool_catalog.py` |
| 任务的对话历史 | `src/data/task_manifest.py` |
| 训练数据的 prompt 生成 | `src/data/preference_builder.py` |
| 一致性检查实现 | `src/training/dpo_dataset.py` |
| 一致性检查入口 | `scripts/prepare/inspect_preferences.py`、`scripts/train/dpo_train.py` |
| 多轮对话的 observation 注入 | `src/models/agent_loop.py` |
| 偏好数据设计（含 prompt 结构说明） | [`docs/method/preference_data_design.md`](../method/preference_data_design.md) |
