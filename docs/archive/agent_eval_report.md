# Agent 多轮执行评测报告

> # ⚠️⚠️ 本报告的全部数字**已作废**（2026-10 复核）
>
> ## 为什么作废
>
> 本报告的数字跑在**旧任务集（v1）**上。v1 的已知缺陷：
> 骨架/任务比只有 **0.52**（「52 种任务换 100 种说法」）、
> 日期**全在过去**（2025-07/08，而项目当前是 2026-10）、
> `multi_tool` 名不符实（期望调用只有 1 个工具）、13 处文本重复、
> `seen`/`unseen` 标签与训练暴露无关。
>
> 另有**评测覆盖缺口**：本报告的 39 个多轮任务**排除了 `flight_search`**，
> 而那正是 SFT 最弱的单类。所以「100%」这个数字不能推广到全部能力。
>
> ## 哪些还能用
>
> * **多轮循环本身的实现**（工具执行 / 观测回灌 / 故障注入 / 轨迹记录）
>   —— 代码事实，不受任务集影响
> * **方法论观察**：例如「故障注入下成功率下降」这个**方向**
> * 不能用：所有具体百分比
>
> ## 下一步
>
> v3 任务集（106 任务，新增 `multi_turn` 类 6 个真多轮任务）已就绪，
> 需在它上面重跑 agent 评测。**在那之前不要引用本报告的数字。**
>
> ---


> **状态**：真实 GPU 实验结果（Qwen2.5-1.5B-Instruct + LoRA，单卡 RTX 3090）。
> 所有数字来自 `experiments/<label>/agent_report.json` 与
> `experiments/<label>/trajectories.jsonl`，可由
> `scripts/eval/run_agent_eval.py` 复现。
>
> **模型版本**：本报告主表用 `outputs/sft_v3/final`（干净数据 + 语料覆盖补全）。
> 早期一版用 `outputs/sft_only/final`（**训练数据含评测任务**，已废弃），
> 两版对照见 §2.3。
>
> ⚠️ **覆盖度局限（必读）**：agent 评测的 39 个多轮任务**不含 `flight_search`
> 与 `weather_advisory`**，而 `flight_search` 恰是单轮评测里最弱的类别（68.2%）。
> 因此 §2 的 100% **只说明「与 agent 循环相关的类别全部过关」，
> 不代表模型整体能力**。见 §3.3。

---

## §0 这份评测回答什么问题

单轮评测（`scripts/eval/run_baseline.py`）把任务里**预计算的 observation**
直接注入 prompt，所以模型是「读现成答案」。它回答的是：

> **给定充足上下文，模型这一步能不能调对工具与参数？**

agent 评测回答另一个问题：

> **不给现成答案，模型能不能自己调工具取信息、读懂返回、再据此决策？**

这两件事**不可互相替代**。本报告的 §4 会用实测数据说明它们的分野。

---

## §1 评测设置

### 1.1 关键设计：不注入预计算 observation

以 `task_0023` 为例，任务定义是：

```
turn0 [user]  帮我订一张2025-07-24 北京到上海的机票，乘客是张伟，经济舱就行
turn1 [tool]  （search_flights 的返回，含 flight_id=MU1000）
turn2 [user]  就订 MU1000 这一班吧
expected      book_flight(flight_id="MU1000", passenger_name="张伟")
```

**`turn2` 里的 `MU1000` 只有模型自己调用 `search_flights` 才能知道。**

| 评测方式 | 播种到 prompt 的内容 | 结果 |
|---|---|---|
| 单轮 | `turn0` + **`turn1` 的 observation** + `turn2` | 模型读答案后直接 `book_flight` |
| **agent** | **只有 `turn0`** | 模型必须自己 `search_flights` → 读返回 → 再 `book_flight` |

agent 循环的时序（实测轨迹）：

```
msg0 [system]      ...
msg1 [user]        帮我订一张2025-07-24 北京到上海的机票，乘客是张伟，经济舱就行
msg2 [assistant]   <tool_call>{"name":"search_flights","arguments":{...}}</tool_call>   ← 模型自己的动作
msg3 [user]        <observation>{"ok":true,"tool":"search_flights","result":{"flights":[{"flight_id":"MU1000",...}]}}</observation>
msg4 [user]        就订 MU1000 这一班吧        ← 拿到结果后才释放
msg5 [assistant]   <tool_call>{"name":"book_flight","arguments":{"flight_id":"MU1000","passenger_name":"张伟"}}</tool_call>
```

### 1.2 工具真的会执行

`ToolRuntime` 包住项目已有的 `execute_tool`，六个工具都有完整执行语义：

| 工具 | 返回示例 |
|---|---|
| `search_flights` | `{"ok":true,"result":{"count":4,"flights":[{"flight_id":"CA5972",...}]}}` |
| `book_flight` | `{"ok":true,"result":{"booking_id":"BKCAF5EF",...}}` |
| `get_weather` | `{"ok":true,"result":{"city":"北京","condition":"雷阵雨","temp_c":32}}` |

失败返回也带有可读的错误语义，模型据此可以纠正：

```
{"ok":false,"error":"未知工具 'nope'。可用工具: ['book_flight','cancel_booking',...]"}
{"ok":false,"error":"参数校验失败: [get_weather.city] missing: 缺少必填参数"}
```

### 1.3 故障注入（测鲁棒性）

真实 agent 面对的工具**会失败**（超时、限流、服务异常）。干净基准里没有故障，
所以必须主动注入才能测出恢复能力。

注入是**确定性**的（由 `hashlib` 从 `(task_id, turn, tool)` 派生，不走随机数），
因此同一任务重跑得到相同故障，指标可复现。

注入采用 **transient 语义**：同一 `(任务, 工具)` 只失败一次，重试会成功 ——
这样测的是「**模型会不会重试**」，而不是「运气好不好」。

### 1.4 任务集

| 项 | 值 |
|---|---|
| 评测任务 | 39 个（`len(turns) > 1`） |
| 其中**真需工具链** | 19 个（`booking_flow` 10 + `multi_tool` 9） |
| 轮数上限 | 5 |
| 采样 | 贪心（T=0.7，每任务 1 条轨迹） |

> **为什么区分「多轮」与「真需工具链」**：`recovery` 与 `cancellation` 的
> `turns` 数组虽有多轮，但目标动作所需信息在首轮或预设 assistant 轮里已具备，
> agent 一轮即可完成。把它们算作「多轮」会稀释 `avg_turns`。
> 判据见 `src/agent/loop.py::task_needs_tool_chaining`。

---

## §2 主结果

**模型**：`outputs/sft_v3/final`（干净训练数据 + 语料覆盖补全，单轮 accuracy 83.00%）

### 2.1 无故障

```bash
python scripts/eval/run_agent_eval.py \
    --adapter outputs/sft_v3/final \
    --label sft_v3_agent --multi-turn-only --max-turns 5
```

| 指标 | 值（39 多轮） | 含义 |
|---|---|---|
| **轨迹成功率** | **100.00%**（39/39） | agent 主指标 |
| 平均轮数 | 1.54 | 效率 |
| 成功轨迹平均轮数 | 1.54 | 越接近 1 越果断 |
| 工具调用有效率 | 98.31% | 仅 1 次无效调用 |
| 放弃率 | 2.56% | |
| 撞上限率 | **0.00%** | 从未卡死 |
| 循环异常率 | **0.00%** | 循环本身无 bug |

按类别：

| 类别 | n | 成功率 | 平均轮数 |
|---|---|---|---|
| `booking_flow` | 10 | **100%** | **2.00** |
| `cancellation` | 8 | **100%** | 1.00 |
| `multi_tool` | 9 | **100%** | **2.00** |
| `recovery` | 12 | **100%** | 1.17 |

### 2.2 真需工具链（严格口径）

`--chain-only` 只保留「后续轮次所需信息只存在于前序工具返回里」的任务
（判据见 `src/agent/loop.py::task_needs_tool_chaining`）。

| 指标 | 值（19 任务） |
|---|---|
| **轨迹成功率** | **100.00%**（19/19） |
| **成功轨迹平均轮数** | **2.00** |
| 工具调用有效率 | **100.00%** |
| 放弃率 | **0.00%** |
| 撞上限率 | 0.00% |

**`avg_turns_to_success = 2.00` 是最强的正确性证据**：真需工具链的任务结构就是
「搜索 → 执行」两步，模型**恰好两步完成，不多不少** —— 说明它既没有跳过
搜索（那会拿不到 `flight_id`），也没有多余空转。

实测轨迹（`task_0023`）：

```
msg1 [user]        帮我订一张2025-07-24 北京到上海的机票，乘客是张伟，经济舱就行
msg2 [assistant]   <tool_call>{"name":"search_flights",...}</tool_call>      ← 自己发起
msg3 [user]        <observation>{"ok":true,...,"flight_id":"MU1000",...}      ← 真实执行结果
msg4 [user]        就订 MU1000 这一班吧                                        ← 拿到结果后才释放
msg5 [assistant]   <tool_call>{"name":"book_flight","arguments":{"flight_id":"MU1000",
                   "passenger_name":"张伟"}}</tool_call>                     ← 恰好 2 轮
```

### 2.3 故障注入（50% 故障率）

```bash
python scripts/eval/run_agent_eval.py \
    --adapter outputs/sft_v3/final \
    --label sft_v3_agent_fault --multi-turn-only --max-turns 5 --fault-rate 0.5
```

| 指标 | 无故障 | **50% 故障** | 变化 |
|---|---|---|---|
| 轨迹成功率 | 100.00% | **89.74%** | −10.3pp |
| **故障恢复率** | — | **77.78%**（14/18） | — |
| 平均轮数 | 1.54 | **2.00** | +0.46 |
| 成功轨迹平均轮数 | 1.54 | 1.71 | +0.17 |
| 撞上限率 | 0.00% | 2.56% | +2.56pp |

按类别（故障下）：

| 类别 | n | 成功率 | 平均轮数 | 无故障时 |
|---|---|---|---|---|
| `booking_flow` | 10 | 90.0% | 2.40 | 100% / 2.00 |
| `cancellation` | 8 | **100%** | 1.25 | 100% / 1.00 |
| **`multi_tool`** | 9 | **66.7%** | **3.11** | **100% / 2.00** |
| `recovery` | 12 | **100%** | 1.33 | 100% / 1.17 |

**三点值得注意**：

1. **鲁棒性可用**：50% 的工具调用会失败，成功率只掉 10.3pp
2. **它真的在重试**：`avg_turns` 从 1.54 升到 2.00 —— 不是硬撑，是重新发起调用
3. **依赖链越长，故障影响越大**：`multi_tool` 从 100% 掉到 66.7%，
   轮数从 2.00 升到 3.11（说明它在反复重试）；而单步的 `cancellation` 仍是 100%。
   这符合「误差累积」的直觉，也说明该指标确实测到了链路长度的影响。

### 2.4 两版模型对照

| 指标 | `sft_only`（**训练数据含评测任务**） | `sft_v3`（干净） |
|---|---|---|
| 39 多轮成功率 | 94.87% | **100.00%** |
| 19 工具链成功率 | 89.47% | **100.00%** |
| 工具调用有效率 | 85.29% | **98.31% / 100%** |
| 平均轮数 | 1.82 | **1.54** |
| 放弃率 | 7.69% | **2.56%** |

> ⚠️ **这张表不能读作「干净数据让模型变好」**：两版的评测任务集相同，
> 但 `sft_only` 的训练数据里混入了评测任务（见消融报告 §8）。
> `sft_v3` 数字更高，是因为**语料覆盖补全**（机场码 13→20、取消任务补
> 确认轮、book 样本只给必填参数）修复了真实的能力缺口。

---

## §3 与单轮评测的对照

### 3.1 两个数字

| | 单轮（100 任务） | agent 多轮（39 任务） |
|---|---|---|
| 主指标 | accuracy **83.00%** | 成功率 **100.00%** |
| 信息 | 含预计算 observation 与参考 assistant 轮 | **无脚手架** |
| 工具是否真执行 | ❌ 只比对参数 | ✅ 真执行 |
| 额外维度 | — | 轮数、故障恢复、工具链 |

### 3.2 agent 评测为何更高

**100% > 83% 看似矛盾，原因是任务集覆盖度不同**：

```
agent 评测的 39 个多轮任务：
  booking_flow 10 · multi_tool 9 · cancellation 8 · recovery 12

单轮评测的 100 个任务：
  上述四类 + flight_search 22 · baggage 15 · weather_advisory 12
```

而 `flight_search` 是单轮评测里**最弱**的类别（68.2%）。它在 agent 评测里
**完全没有出现**（它的 `turns` 只有单轮，不满足 `len(turns) > 1`）。

### 3.3 ⚠️ 必须声明的覆盖度局限

**本报告的 100% 不能读作「模型全面过关」。** 它只覆盖了与 agent 循环相关的
四类任务，且这四类在单轮评测中本来就是模型表现最好的部分：

| 类别 | 单轮 accuracy | 是否在 agent 评测里 |
|---|---|---|
| `weather_advisory` | 100% | ❌ 否（单轮） |
| `baggage` | 100% | ❌ 否（单轮） |
| `multi_tool` | 100% | ✅ 是 |
| `recovery` | 91.7% | ✅ 是 |
| `booking_flow` | 72.2% | ✅ 是 |
| `cancellation` | 66.7% | ✅ 是 |
| **`flight_search`** | **68.2%** | ❌ **否** |

**准确表述**：「在与 agent 循环相关的四个类别上，模型 100% 成功；
但最弱的 `flight_search` 类不在此评测范围内。」

若要把 agent 评测扩展到全部任务，需要给 `flight_search` 类任务构造多轮结构
（例如「先搜索、用户看到结果后再追问筛选条件」），这是后续工作。

---

## §4 两个必须解释清楚的数字

### 4.1 工具调用有效率：一个已消失的旧问题

`sft_only` 版（**训练数据含评测任务**）的工具有效率只有 **85.29%**，
10 次失败全是 `cancel_booking 缺少 reason`：

```
task_0056  cancel_booking({'booking_id': 'BK9A4C6F'})                      ok=False
           cancel_booking({'booking_id': 'BK9A4C6F', 'reason': '行程变更'})  ok=True
task_0057  ... 'reason': '身体不适'
task_0058  ... 'reason': '会议取消'
```

**sft_v3 版已经是 98.31%（39 多轮）/ 100.00%（19 工具链）**，该问题消失。

**成因**：agent 模式下工具**真的执行**，`execute_tool` 的 schema 校验生效；
而单轮评测只比对 `graded_args` 声明的参数（`cancel_booking` 只评 `booking_id`），
**永远不会触发 schema 校验**。两者口径不同：

| 评测方式 | `graded_args` 未覆盖的必填参数 | 后果 |
|---|---|---|
| 单轮 | 不检查 | 漏 `reason` 也能通过 |
| **agent** | **检查**（真实执行） | 被拒 → 迫使模型补全 |

**这个差异曾两次判断错误**：

1. 我先认为「单轮数字因此虚高」—— **错**。实测 `sft_only` 与 `baseline`
   在这 9 个任务上 **0/4 次漏参数**，单轮数字并未虚高。原因是单轮 prompt 里
   含预设 assistant 轮（把 `reason` 显式复述了一遍），**等于给了脚手架**。
2. 我随后认为评分器的 `extra_fields` 没有 schema 白名单会**压低**准确率 ——
   **也错**。`extra_fields` 只设诊断子类型标签，`passed` 只取决于
   `category == SUCCESS`，而多给字段时 `category` 恒为 `SUCCESS`。
   实测重评分后所有数字**一字未变**。

**两点结论**：
* **「同一份答案在两个评测下不一致」时，要先查两个评测各给了什么上下文**，
  而不是直接断定其中一个虚高。
* **判断一个字段是否影响打分，要看 `passed` 的赋值处**，
  不能从「它被列在错误报告里」推断 —— 诊断标签与判定逻辑是两条路径。

（`extra_fields` 的 schema 白名单修复仍然保留：它让诊断报告不再冤枉模型，
虽然不影响任何准确率数字。）

### 4.2 评测任务集与训练语料的口径差异（保留）

`cancel_booking` 的 schema 要求 `booking_id` + `reason` 两个必填参数，
但评测任务里 `expected.args` **只有 `booking_id`** —— 实测 **9 个任务**如此：

| 任务 | 工具 | schema 必填但未评分 |
|---|---|---|
| `task_0056` ~ `task_0063` | `cancel_booking` | `reason` |
| `task_0097` | `book_flight` | `flight_id` |

**这是有意为之的**：`reason` 是自由文本（不该逐字评分）、`flight_id` 需要模型
自己搜索获取（不该要求它凭空知道）。所以评分只比对 `booking_id` /
`passenger_name`，而**必填校验交给工具执行层** —— 这恰好使 agent 评测
比单轮评测更严格。

### 4.3 评测设置上的一个坑

开发 agent 循环时，播种逻辑改错了方向，导致指标剧烈偏移：

| 播种策略 | 成功率 | `booking_flow` | `cancellation` |
|---|---|---|---|
| **只播 `turn0`（正确）** | **94.87%** | 80.0% | 100% |
| 播所有 user 轮 + assistant 轮（错误） | **53.85%** | **0%** | **0%** |

> ⚠️ **这张表的两个数字都出自「训练数据含评测任务」的那一版模型**
> （`outputs/sft_only/final`），因此**绝对值不可引用**。
> 它在这里只用来**说明播种方向错误会造成的量级**（41pp）。
>
> 干净版模型（`sft_v3`）在这 39 个任务上两种播种都是 100% ——
> 说明**该对照在当前模型上已无法复现**（模型能力足以覆盖播种差异）。
> 保留这一节是因为**推理与测试本身仍然有效**：
> `test_agent_seeding_does_not_leak_future_turns` 正反两个方向都断言，
> 与模型强弱无关。

**错在哪**：

1. **播了 `turn2`（「就订 MU1000 这一班吧」）** —— 那是模型动作的**结果**，
   提前泄露等于给答案。模型跳过搜索直接下单 → 10 条 `booking_flow` 全失败。
2. **播了 20 个 assistant 预设轮** —— 那是**目标动作的参考输出**，
   不是模型已看到的输入。播进去造成「模型没说过那句话，用户却回应了它」
   的不连贯 → 8 条 `cancellation` 全失败。

**教训**：agent 评测的播种必须严格对齐「评测发生在对话的哪个时刻」。
这个推理已固化在 `src/agent/loop.py::run()` 的注释与
`tests/test_agent.py::test_agent_seeding_does_not_leak_future_turns` 里
（正反两个方向都断言）。

---

## §5 指标口径（避免混用）

| 指标 | 定义 | 主指标？ |
|---|---|---|
| `success_rate` | 整条轨迹达成期望动作的比例 | ✅ agent 主指标 |
| `fault_recovery_rate` | **注入故障的轨迹**中最终成功的比例（分母是故障轨迹数） | ✅ 鲁棒性主指标 |
| `avg_turns` | 平均轮数（效率） | |
| `avg_turns_to_success` | 成功轨迹的平均轮数 | |
| `tool_call_precision` | 1 − 模型自身无效调用 / 总调用 | |
| `give_up_rate` | 提前给最终回答的比例 | |
| `max_turns_rate` | 撞轮数上限的比例 | |
| `error_rate` | 循环内部异常比例（应为 0） | |

⚠️ **`fault_recovery_rate` 的分母是「注入过故障的轨迹数」，不是全部轨迹。**
若用全部轨迹做分母，故障少时该比率会被稀释得看不出问题。

⚠️ **`num_failed_calls` 与 `num_invalid_calls` 必须区分**：
前者含注入故障，后者只算模型自身错误。混用会把「环境抖动」算成「模型犯错」。

---

## §6 复现方式

```bash
# 无故障
python scripts/eval/run_agent_eval.py \
    --adapter outputs/sft_v3/final \
    --label sft_agent_v3 --multi-turn-only --max-turns 5

# 50% 故障注入
python scripts/eval/run_agent_eval.py \
    --adapter outputs/sft_v3/final \
    --label sft_agent_fault_v3 \
    --multi-turn-only --max-turns 5 --fault-rate 0.5

# 只要真需工具链的 19 个任务（需在脚本里用 task_needs_tool_chaining 过滤）
python scripts/eval/run_agent_eval.py \
    --adapter outputs/sft_v3/final --label sft_agent_chain
```

产物：

| 文件 | 内容 |
|---|---|
| `experiments/<label>/agent_report.json` | 聚合指标 + 元数据 |
| `experiments/<label>/trajectories.jsonl` | 每条轨迹的完整记录（可回放） |

**轨迹可直接用于学习失败模式**：

```bash
python - <<'PY'
import json
from pathlib import Path
rows = [json.loads(l) for l in Path("experiments/sft_agent_v3/trajectories.jsonl").open(encoding="utf-8") if l.strip()]
for r in rows:
    if not r["success"]:
        print(r["task_id"], r["category"], r["verdict"])
PY
```

---

## §7 未做与下一步

| 项 | 状态 | 说明 |
|---|---|---|
| 多轮任务集的「真工具链」子集单独评测 | 未做 | 需给 `run_agent_eval.py` 加 `--chain-only` 开关 |
| 按 `assistant_turns_preloaded` 分组对照 | 未做 | 当前播种为 0，该字段恒为 0，保留是为了未来兼容 |
| DPO 模型在 agent 循环上的评测 | 未做 | DPO 从 SFT 续训的那一版崩塌（见消融报告），无可用 adapter |
| 多次采样与 pass^k | 未做 | 当前每任务 1 条轨迹；agent 多轮下采样成本高 |
| 故障率扫描（10%/30%/50%） | 未做 | 可给出「成功率 vs 故障率」曲线 |
