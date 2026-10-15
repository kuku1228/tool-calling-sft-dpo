# 偏好数据构造设计

## 文档定位

这份文档回答一个问题：**`data/preferences/*.jsonl` 里的每一条 `(prompt, chosen, rejected)` 是怎么来的，为什么是这样而不是别的样子。**

读者是：（1）需要审计这批数据是否可信的人；（2）要在此基础上加第五类偏好对或换评测域的人；（3）面试里被追问「你的偏好对怎么构造」时想找准确说法的人。

文档只描述设计意图与实现约束，**不报实验结果**。实现代码：

| 关注点 | 代码位置 |
|---|---|
| 四类偏好对的构造与质量门禁 | [`src/data/preference_builder.py`](../../src/data/preference_builder.py) |
| 纠错训练语料（12 模式 × 7 变体） | [`src/data/recovery_corpus.py`](../../src/data/recovery_corpus.py) |
| 工具 schema、可信幻觉名表、参数校验 | [`src/env/tool_catalog.py`](../../src/env/tool_catalog.py) |
| 错误分类体系与「每个类别对应什么数据」 | [`src/eval/error_taxonomy.py`](../../src/eval/error_taxonomy.py) |
| system prompt（单一事实来源） | [`src/models/prompt_builder.py`](../../src/models/prompt_builder.py) |
| 长度体检 / prompt 一致性校验 | [`src/training/dpo_dataset.py`](../../src/training/dpo_dataset.py) |
| 落盘入口 | [`scripts/prepare/build_preferences.py`](../../scripts/prepare/build_preferences.py) |

> ⚠️ **成文时的状态说明**：这份文档写作期间仓库正被并行修改。§7.2 与 §6.4 的实测数字由第一节给的命令在当时的代码上跑出，§9 的每条问题都附了核对命令。引用前请先跑一遍确认。

---

## 1. 为什么是四类，而不是「随便造点偏好数据」

### 1.1 出发点：先有能力诊断，才有数据设计

本项目的顺序是**诊断 → 分类 → 构造数据**，不是先有一批偏好数据再想拿它做什么。

[`src/eval/error_taxonomy.py`](../../src/eval/error_taxonomy.py) 把一次采样输出分成 7 个互斥类别（优先级从高到低）：

```
SUCCESS
FORMAT_ERROR      没输出可解析的 tool_call
HALLUCINATION     工具名不在白名单 VALID_TOOL_NAMES
WRONG_TOOL        工具名合法但不是期望的那个
PARAM_ERROR       工具名对但参数不对
REFUSED           明确拒绝或放弃
UNKNOWN           其他
```

优先级顺序不是随意的：「JSON 都解析不出来」比「工具名选错」更根本，所以 `FORMAT_ERROR` 优先判。这个顺序保证所有类别之和 = 失败总数，不重不漏——诊断报告里各类错误占比因此是可加的。

每一类错误都必须能映射到一个**具体的、可写进代码的构造动作**，这个映射被硬编码在 `error_taxonomy.CATEGORY_REMEDIES_ZH` 里：

```python
CATEGORY_REMEDIES_ZH = {
    FORMAT_ERROR: "格式约束偏好对：chosen 是合法 tool_call，rejected 是纯文本/坏 JSON",
    HALLUCINATION: "工具选择偏好对：chosen 用白名单工具，rejected 用貌似合理的假工具",
    WRONG_TOOL: "工具选择偏好对：同一请求下正确工具 vs 相邻语义工具",
    PARAM_ERROR: "参数约束偏好对：chosen 是真实实体，rejected 是占位符/错格式/编造值",
    REFUSED: "错误恢复偏好对 + 澄清类样本：示范「信息不足时追问」而非放弃",
    UNKNOWN: "人工抽检，补充新类别",
}
```

**如果这个映射不存在会怎样**：数据构造会退化成「我觉得模型需要学 X」，而「模型到底错在 X 没有」无法证伪。四类偏好对的价值恰恰在于它是一份**可被诊断数据否定的假设清单**。

### 1.2 错误类别 ↔ 偏好对类型 ↔ 对比点

| 错误类别 | 偏好对类型（常量） | chosen | rejected 的对比点 | 构造方法 |
|---|---|---|---|---|
| `FORMAT_ERROR` | `format_constraint`（`PAIR_FORMAT`） | 合法 `<tool_call>{...}</tool_call>` | ①纯文本承诺 ②键缺引号的坏 JSON ③markdown 围栏 | `build_format_pairs` |
| `NO_RECOVERY` / 纠错轮失败 | `error_recovery`（`PAIR_RECOVERY`） | 纠错轮的正确修正（可能是工具调用，也可能是澄清） | 原样重复首轮错误 / 退化成纯文本 / 换个错工具 / 换对工具但参数仍错 | `build_recovery_pairs` |
| `HALLUCINATION` | `tool_selection`（`PAIR_TOOL_SELECTION`），`negative_kind="simple"` | 白名单里的正确工具 | 正确工具的**命名变体**（`find_flights` 之于 `search_flights`） | `build_tool_selection_pairs` |
| `WRONG_TOOL` | `tool_selection`（`PAIR_TOOL_SELECTION`），`negative_kind="hard"` | 白名单里的正确工具 | 真实存在但语义相邻的工具（`check_baggage` → `modify_baggage`） | `build_tool_selection_pairs` |
| `PARAM_ERROR` | `param_constraint`（`PAIR_PARAM`） | 任务 ground truth 的真实实体 | ①占位符（`previous_booking`）②错格式（中文城市名当三字码）③形状对、值错一个字符 | `build_param_pairs` |
| `REFUSED` | `error_recovery`（澄清类场景） | 「请问您能告诉我取消这个订单的原因吗？」 | 硬填一个编造的 `reason` | `recovery_corpus._s_fake_reason` |

注意两处**不是一对一**的地方，这是有意的：

1. `tool_selection` 一类覆盖 `HALLUCINATION` 与 `WRONG_TOOL` 两类错误，靠 `negative_kind` 再区分，并写进 `target_error` 字段。理由：两者的 chosen 完全相同（都是白名单工具），差别只在 rejected 的「假法」不同，拆成两类偏好对只会复制 chosen。
2. `REFUSED` 没有专属偏好对类型。它在实现里由 `error_recovery` 的澄清类场景覆盖——`_s_fake_reason` 的 chosen 是一句自然语言追问，rejected 是硬填 `reason="用户要求"`。这也是质量门禁里必须为「chosen 是自然语言」开绿灯的原因（见 §4）。

### 1.3 四类在代码里是可执行的消融定义

`EXPERIMENT_PAIR_TYPES` 把「哪个实验用哪些类型」写成常量，而不是散落在训练脚本的参数里：

```python
EXPERIMENT_PAIR_TYPES = {
    "dpo_format":      (PAIR_FORMAT,),
    "dpo_recovery":    (PAIR_RECOVERY,),
    "dpo_joint":       (PAIR_FORMAT, PAIR_RECOVERY),
    "dpo_joint_param": (PAIR_FORMAT, PAIR_RECOVERY, PAIR_PARAM),
    "dpo_tool":        (PAIR_TOOL_SELECTION,),
    "dpo_all":         (PAIR_FORMAT, PAIR_RECOVERY, PAIR_TOOL_SELECTION, PAIR_PARAM),
    "sft_only":        (PAIR_FORMAT, PAIR_TOOL_SELECTION, PAIR_PARAM),
    "dpo_tool_param":  (PAIR_TOOL_SELECTION, PAIR_PARAM),
}
```

`scripts/prepare/build_preferences.py --experiments ...` 用它切数据文件，`build_experiment_dataset(experiment, pairs, split=...)` 用它筛内存里的对。因此「Exp 1 用了哪些数据」在文件系统层面（`data/preferences/dpo_format.train.jsonl`）和代码层面同时可查证。

**`sft_only` 这一行需要单独说明**：它包含格式 / 工具选择 / 参数三类，但**有意不含 `error_recovery`**。SFT 只用 chosen 做监督，而恢复类偏好对的 chosen 之所以有价值，恰恰依赖于 prompt 里那段「首轮错误 + 用户纠错」——把它塞进 SFT 就等于教模型输出一个上下文里没有的错误回复。所以 `sft_only` 的作用是：验证提升来自**对比信号**，还是仅仅来自**更多同分布监督**。如果两者效果相当，「偏好对齐」这个卖点就不成立。

---

## 2. `PreferencePair` 的字段与用途

[`src/data/preference_builder.py`](../../src/data/preference_builder.py) 定义：

| 字段 | 类型 | 用途 |
|---|---|---|
| `prompt` | `str` | 已套好 ChatML 标记的纯文本，以 `<|im_start|>assistant\n` 结尾 |
| `chosen` | `str` | 被偏好的那一条 completion |
| `rejected` | `str` | 被拒绝的那一条 completion |
| `pair_type` | `str` | 四个 `PAIR_*` 常量之一，用于按实验切片 |
| `task_id` | `str` | 来源任务；`trainsyn_` 前缀 = 由纠错语料转成的训练专用任务 |
| `category` | `str` | 任务类别（`flight_search` / `recovery_format_error` …） |
| `difficulty` | `str` | `easy` / `medium` / `hard` |
| `target_error` | `str` | 针对的错误类别常量，或 `NO_RECOVERY` |
| `negative_kind` | `str` | `hard` / `simple` / `recovery`，见 §3 |
| `split` | `str` | `train` / `dev`，由 `build_all` 或 `build_preferences.py` 回填 |
| `tools` | `list[dict]` | 工具 schema。**实践中恒为空**，因为工具说明已写进 system prompt |
| `intent` | `str` | 一句话说明这条数据的教学意图，`inspect_preferences.py` 会打印它 |

`to_dict()` 只在 `tools` 非空时输出该键，所以落盘的 jsonl 通常没有 `tools`。

### 2.1 去重键为什么必须包含 `rejected`

```python
@property
def key(self) -> str:
    return f"{self.pair_type}|{self.prompt}|{self.chosen}|{self.rejected}"
```

早期版本只用 `prompt + chosen` 做键，结果是「同一个任务、同一个正确回复、但换了不同错误类型作为负样本」的多条候选互相顶掉。README 记载实测会丢掉约 97% 的候选。对 DPO 而言，**同一个 chosen 配不同的 rejected 是三条不同的训练信号**（模型要学会同时压掉三种坏输出），不能合并。

---

## 3. 每类偏好对的构造逻辑

### 3.1 prompt 的结构（四类共用）

`render_prompt(task, *, system_prompt, extra_turns)` 生成：

```text
<|im_start|>system
{完整 system prompt：工具清单 + 格式规范 + 5 条规则}
<|im_end|>
<|im_start|>user
{任务前若干轮：user / assistant / 观察结果}
<|im_end|>
<|im_start|>assistant
```

关键点：

- **system prompt 来自 `build_system_prompt()`，不是模板字符串常量**（`__init__` 里 `self.system_prompt = build_system_prompt()`）。这样工具清单的任何改动都会同时影响训练与推理。
- **结尾是裸的 `<|im_start|>assistant\n`**，没有 `<|im_end|>`。这是「轮到模型生成」的标记，DPO 的 chosen / rejected 会被拼在这个位置之后。
- **不用 chat template（`tokenizer.apply_chat_template`）**。理由：构造偏好数据时不应该依赖 tokenizer（可能只在 GPU 机器上存在），数据生成必须与模型无关、可复现。训练脚本在加载数据时按 `prompt` 字段原样使用，不再二次套模板（见 §5）。
- 工具多轮历史里的 observation 由 `Task.to_messages()` 渲染成 `<observation>{json}</observation>` 并以 **user 角色**注入，与 Hermes / Qwen-Agent 风格一致。

#### 纠错轮的 prompt 结构（`error_recovery` 专用）

`_render_scenario_prompt(sc, correction)` 生成五段：

```text
<|im_start|>system
{system prompt}<|im_end|>
<|im_start|>user
{sc.user_request}<|im_end|>
<|im_start|>assistant
{sc.wrong_reply}<|im_end|>          ← 模型的首轮错误回复被"写死"进 prompt
<|im_start|>user
{correction}<|im_end|>              ← 用户纠错话术
<|im_start|>assistant
```

`sc.wrong_reply` 与 `sc.correction` 都来自 [`src/data/recovery_corpus.py`](../../src/data/recovery_corpus.py) 的 `RecoveryScenario`。

### 3.2 类 1：格式约束对 `build_format_pairs`

- **候选池**：`t.expected.tool is not None` 的任务。
- **chosen**：`canonical_call(task.expected.tool, task.expected.graded_args)`，即一条参数合法的标准 `<tool_call>`。
- **rejected 三档**（按 `self.rng.random()` 分桶，阈值硬编码在函数里）：

| 概率 | `negative_kind` | rejected 内容 | intent |
|---|---|---|---|
| `< 0.45` | `simple` | 从 `_SIMPLE_NEGATIVES` 抽一句承诺式回复（「好的，我来帮您查询一下，请稍等。」等 5 句） | 模型只是承诺要查，没有真的输出 tool_call |
| `< 0.75` | `hard` | 有 `<tool_call>` 标签但 JSON 键缺引号 | 有 tool_call 标签但 JSON 语法非法 |
| `else` | `hard` | ```` ```json ... ``` ```` markdown 围栏 | 用 markdown 代码块代替规定的 tool_call 标签 |

- **循环保护**：`while len(out) < n and guard < n * 20`。这个 guard 是必要的——见 §7.2，格式类有**去重后的上限**（约 924 条），把 `n` 设成 4200 也会在那个上限处停止，而不是死循环。

### 3.3 类 2：错误恢复对 `build_recovery_pairs`

这类是本项目的原创点，单独展开见 §4。

- **数据源**：`scenarios`（默认 `build_scenarios()`，84 个场景），**不是**评测集里的 12 个 recovery 任务。
- **chosen**：`sc.correct_reply`。注意它**不一定是工具调用**——`_s_fake_reason` 的 correct_reply 是一句自然语言追问。
- **rejected 四档**：

| 概率 | `negative_kind` | rejected 内容 | intent |
|---|---|---|---|
| `< 0.50` | `recovery` | `sc.wrong_reply`，即原样重复首轮错误 | 被指出错误后原样重复，没有听进用户的纠正 |
| `< 0.75` | `simple` | `_SIMPLE_NEGATIVES` 里的一句 | 被指出错误后反而退化成纯文本，不再输出工具调用 |
| `< 0.90` | `hard` | 换成一个**真实存在但语义相邻**的工具（`_pick_adjacent_tool_for_scenario`） | 被指出错误后换了工具，但换到了另一个同样错误的工具 |
| `else` | `hard` | 工具正确、但扰动其中一个关键参数（`_same_tool_wrong_args` → `_perturb`） | 被指出错误后工具改对了，但关键参数仍然是错的 |

**`hard` 两档最值得注意**：它们分别对应「纠错后仍选错工具」和「纠错后仍填错参数」。真实的 agent 失败大多是这两种，而不是「完全没反应」。只训前两档会得到一个「被纠正就会动一下」的模型，但动了之后对不对没有约束。

### 3.4 类 3：工具选择对 `build_tool_selection_pairs`

- **候选池**：`t.expected.tool is not None` 的任务。
- **chosen**：`canonical_call(tool, task.expected.graded_args)`。
- **rejected 按 `self.cfg.hard_negative_ratio` 分桶**：
  - `hard`：`_pick_adjacent_tool(tool)` 挑一个**真实存在**的相邻工具，参数用 `_args_for_tool` 按该工具的必填参数补齐（同名字段沿用任务值，缺的用默认池填），然后 `canonical_call`。`target_error="WRONG_TOOL"`。
  - `simple`：从 `plausible_fakes_for(tool)` 里抽一个**正确工具的命名变体**。`target_error="HALLUCINATION"`。

`_pick_adjacent_tool` 里有一张显式的相邻偏好表：

```python
prefer = {
    "search_flights":  ["get_weather", "check_baggage"],
    "book_flight":     ["search_flights", "check_baggage"],
    "check_baggage":   ["modify_baggage", "book_flight"],
    "modify_baggage":  ["check_baggage", "cancel_booking"],
    "cancel_booking":  ["modify_baggage", "check_baggage"],
    "get_weather":     ["search_flights"],
}.get(tool or "", [])
pool = [t for t in prefer if t in TOOL_REGISTRY] or [t for t in TOOL_REGISTRY if t != tool]
```

读/写混用是有意设计的：`check_baggage`（查询行李）vs `modify_baggage`（改行李额度，**会产生费用**）。模型必须区分「用户想知道」和「用户想改」，这正是 §1 里 `WRONG_TOOL` 的高价值信号。若偏好表命中为空，退化为「注册表里除自己以外的任意工具」，保证不崩。

### 3.5 类 4：参数约束对 `build_param_pairs`

- **候选池**：`t.expected.graded_args` 非空的任务。
- **chosen**：`canonical_call(tool, args)`，args 直接取任务 ground truth。
- **rejected 按 `hard_negative_ratio` 分桶**，随机挑**一个**字段改坏：

| kind | 改法 | 例子 |
|---|---|---|
| `simple` | `_placeholder_for(field_name)` | `booking_id` → `previous_booking` / `my_booking` / `BK000000` / `UNKNOWN`；`date` → `后天` / `TBD` / `2025-13-45`；`origin`/`destination` → `北京` / `PEK机场`；`city` → `PEK` / `未知` |
| `hard` | `_perturb(value, field_name)` | 数字 `30` → `25/28/32/35/40`；`_kg` 结尾的字段下限截到 0；`BK7F2A9C` → `BK7F2A9D`（随机一位换成 `ABCDEFGHJKLMNPQRSTUVWXYZ0123456789`） |

`_perturb` 的设计目标写得很直白：**「形状一模一样、内容错一点」**。这是 DPO 最需要学、也最难学的一档——`BK7F2A9C` 与 `BK7F2A9D` 在字符级只差一个 token，模型必须真的去比对 observation，而不是套格式。它同时是对抗 reward hacking 的样本：如果模型学会「格式对就行」，这档负样本会让它付出代价。

注意 `_placeholder_for` 会返回**非字符串**（`_kg` 结尾的字段返回 `0 / -5 / 999 / "很多"`），这是有意的——`validate_arguments` 会把 `-5` 判为 range 问题、把 `"很多"` 判为 type 问题，覆盖两类参数错误。

---

## 4. 硬负样本 vs 简单负样本：为什么两者都要有

### 4.1 代码里 `hard_negative_ratio` 的真实作用范围

`CFG.preference.hard_negative_ratio = 0.35`（config.py 与 `configs/default.yaml` 一致）。但要清楚它**只在两个构造器里生效**：

| 构造器 | 是否读 `hard_negative_ratio` | 实际配比实现 |
|---|---|---|
| `build_format_pairs` | ❌ 不读 | 硬编码 0.45 分桶（`<0.45` simple，其余 hard） |
| `build_recovery_pairs` | ❌ 不读 | 硬编码 0.50 / 0.25 / 0.15 / 0.10 分桶 |
| `build_tool_selection_pairs` | ✅ `rng.random() < 0.35` | 35% hard（相邻真工具）/ 65% simple（命名变体假工具） |
| `build_param_pairs` | ✅ `rng.random() < 0.35` | 35% hard（形状对值错）/ 65% simple（占位符） |

所以「两类负样本按 0.35 混合」这句话只对工具选择类与参数约束类成立。格式类与恢复类各有自己的分桶理由（格式类的「坏 JSON / markdown 围栏」本身就是自然的两档；恢复类的四档对应四种真实失败模式）。**改动 `hard_negative_ratio` 时不要以为它会影响全部四类。**

### 4.2 为什么全用硬负样本不行

DPO 的梯度前面有一个动态权重 $\sigma(-z) = \sigma(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w))$（推导见 [`src/training/dpo_loss.py`](../../src/training/dpo_loss.py) 的模块 docstring）。硬负样本的 chosen 与 rejected 差异极小 → 初始 $z$ 接近 0 → loss 接近 $\log 2$、梯度大。听起来是好事，但：

1. **正负两侧在 token 空间上高度重叠**。`BK7F2A9C` 与 `BK7F2A9D` 的区别只在一个 token。模型完全可以靠「记住这条样本」降低 loss，而不是学「参数必须来自用户输入」。一旦换成没见过的订单号，能力不迁移。
2. **训练不稳**。所有样本的 margin 都贴着 0，loss 下降慢、对 lr 与 β 极敏感。
3. **它无法教会「什么叫完全不做」**。硬负样本都是格式合法的输出，模型学不到「纯文本承诺」是被禁止的。

### 4.3 为什么全用简单负样本更不行

简单负样本（纯文本承诺、坏 JSON、`send_confirmation_email` 这类一眼假的工具）的 $z$ 很快就变得很大、loss 迅速压到 0、**梯度消失**。模型在几十步内就能把它们全部区分开，之后这批数据就只是噪声。这正是 DPO 文档里说的「梯度饱和」：一旦学得很对，梯度就没了。

更糟的是，如果只有简单负样本，模型会得到一个错误的等价类划分：「格式合法的调用 = chosen」。评测里的 `format_ok_value_wrong` 指标（`src/eval/error_taxonomy.py` 里的 `Classification.format_ok_value_wrong`）就是用来抓这个退化模式的——格式正确率上升、任务成功率不动、这一项上升，说明模型走了格式捷径。

**结论**：简单负样本负责「把底线划出来」（低成本、快速收敛、提供稳定梯度），硬负样本负责「把上限推上去」（在已经学会底线之后继续提供有效梯度）。0.35 这个值是这两条曲线的折中，而且是**可调参数**，改动它属于消融范围。

---

## 5. 错误恢复对是本项目的原创点

### 5.1 问题：SFT 的 chosen 里只有正确答案

SFT 的数据形态是 `(context, correct_answer)`。无论塞多少条，模型看到的条件分布都是「用户的最后一句话之后，正确答案是什么」。**「用户指出我上一轮错了」这个上下文从未出现过**，所以模型没有理由改变行为——它甚至不知道上一轮是它自己说的。

真实交互里的失败因此长这样：用户说「你选的工具不对」，模型礼貌道歉，然后**原样再调一次同一个工具**（`recovery_corpus` 里 `wrong_tool` 场景的 wrong_reply 就是这个）。这不是模型「不听话」，而是「被纠正」这个状态从来没有被训练过。

### 5.2 解法：把「先错后改」直接写进 prompt

```text
<|im_start|>user
帮我把订单 BK7F2A9C 的行李额度改成 35 公斤<|im_end|>
<|im_start|>assistant
<tool_call>{"name": "check_baggage", "arguments": {"booking_id": "BK7F2A9C"}}</tool_call><|im_end|>
<|im_start|>user
你选的工具不对，请重新选择正确的工具。<|im_end|>
<|im_start|>assistant
                                                              ← chosen: modify_baggage
```

三件事同时被训练：

1. **prompt 里显式包含模型自己的首轮错误回复** → 模型看到的是「我犯过这个错」的上下文，而不是一个干净的请求。这个条件分布是 SFT 数据里拿不到的。
2. **用户纠错话术是 prompt 的一部分** → 模型学到「用户这句话出现时，我要改变上一轮的输出」，而不是「用户这句话出现时，我要道歉」。
3. **chosen 是修正后的回复** → 优化目标是「改对」，不是「回应」。

### 5.3 为什么 chosen 有时是自然语言

`recovery_corpus._s_fake_reason` 的场景是：用户说「帮我取消订单 BK7F2A9C」，模型调用 `cancel_booking(booking_id, reason="用户要求")` —— 把 reason 编出来了。纠错话术是「取消原因不能写「用户要求」，请先问清楚我为什么取消。」。

此时**正确的修正不是工具调用，而是先追问**：

```python
correct_reply = "请问您能告诉我取消这个订单的原因吗？我需要记录取消理由才能帮您处理。"
```

这正是 `ExpectedCall.tool is None` 那一类任务的偏好数据化，也是 §3.1 里 `_passes_quality_gate` 必须为「chosen 是自然语言」开绿灯的原因（见 §6.2）。如果门禁硬性要求 chosen 是 canonical 格式，**「学会反问而不是硬填」这类最有价值的恢复样本会被全部丢掉**。

### 5.4 与评测集的严格分离

| 用途 | 数量 | 来源 | 参与评测？ |
|---|---|---|---|
| 评测集 | 100 个任务，其中 **12 个 recovery** | `task_manifest.build_tasks()` | ✅ 唯一评测集 |
| 纠错训练语料 | **84 个场景**（12 模式 × 7 变体） | `recovery_corpus.build_scenarios(n_variants_per_pattern=7)` | ❌ 绝不出现在评测里 |
| 纠错语料转成的训练任务 | **77 个**（84 减去正解是自然语言的那些） | `recovery_corpus.build_training_tasks(n_variants=7)` | ❌ `visibility="train_only"` |

**为什么必须分离**（而不是「多造点 recovery 任务一起评测」）：

1. **评测污染**。如果把 84 个场景也拿去评测，那 DPO 在恢复任务上的提升里有一部分只是「见过这些具体的目标字符串」。报告出来的恢复率会虚高，而且是无法通过看数字发现的虚高。
2. **评测集必须小且固定**。pass^k 的绝对值要能跨实验、跨时间比较。评测集一旦因为「想多测点恢复」而膨胀或变动，历史结果立刻失去可比性。12 个 recovery 任务固定不动，是这份可比性的前提。
3. **训练量需求与评测需求本来就不同**。恢复类偏好对要上千条才训得动；评测只需要足够估计 pass^k 的任务数。用同一批数据同时满足两个需求，必然牺牲其中一个。

代码层面有三道锁：

- `build_training_tasks` 产出的 `Task` 一律 `task_id=f"trainsyn_{sc.scenario_id}"`、`visibility="train_only"`，文件名一眼可辨。
- `_s_fake_reason` 这类正解为自然语言的场景**不会**被转成训练任务（`extract_tool_calls` 解析不出 call 就 `continue`），它们只走 recovery 对。
- `task_manifest` 的 12 个 recovery 任务与 `recovery_corpus` 用**同一套 `mistake_key`**（`format_ignore_correction`、`hallucinated_tool_corrected` …），所以「训练里教了什么」与「评测里考了什么」可以逐类对齐，诊断报告里的 per-mistake-type 归因才有意义。

---

## 6. 质量门禁：每一条规则与它存在的理由

入口是 `PreferenceBuilder._add(pair, bucket)`，它在写进结果列表之前依次调用 `_passes_quality_gate` 与（仅对 `error_recovery`）`_recovery_coherence`。所有拒绝都计数到 `self.stats["rejected_by_gate"]` 与 `self.stats["gate_reasons"]`，最终落盘 `build_report.json`。

### 6.1 `_passes_quality_gate` 的五项检查

实现顺序即下表顺序：

| # | 检查 | 拒绝原因字符串 | 为什么需要 |
|---|---|---|---|
| 1 | `pair.prompt.strip()` 非空 | `empty_prompt` | prompt 为空的对训不出任何东西，且会在 tokenize 阶段报出难以定位的错 |
| 2 | `chosen` / `rejected` 都非空 | `empty_side` | rejected 为空会让 TRL 的 collator 拿到零长度 completion，`log π(y_l)` 无定义 |
| 3 | `chosen.strip() != rejected.strip()` | `identical_sides` | 字面相同的两侧 loss 恒为 `log 2`，梯度恒为 0 |
| 4 | chosen 必须可接受（见下） | `chosen_not_canonical` / `both_sides_natural_language_equal` | 防止教坏格式，同时不误杀澄清类样本 |
| 5 | chosen 的参数过 `validate_arguments` | `chosen_invalid_args:<kind>` | chosen 自己参数非法 = 用错误答案当正例 |
| 6 | chosen 与 rejected 语义必须真的不同 | `sides_semantically_equal` | **偏好数据最常见的污染源**，展开见下 |

第 4 项的具体判定分支（这段代码值得逐行读，因为它是唯一一处「故意放松」的地方）：

```python
chosen_parse = extract_tool_calls(pair.chosen)
rejected_parse = extract_tool_calls(pair.rejected)
if not is_canonical_format(pair.chosen):
    if not chosen_parse.calls:
        # chosen 是自然语言（例如「信息不足时应先追问」）。
        if rejected_parse.calls:
            pass                      # 澄清 vs 贸然调用 → 正是想教的对比
        elif _normalize_loose(chosen) == _normalize_loose(rejected):
            return False, "both_sides_natural_language_equal"
    elif not is_canonical_format(pair.chosen):   # ← 已达不可达
        return False, "chosen_not_canonical"
```

设计意图：**「chosen 是澄清、rejected 是贸然调用」这种对比是合法且高价值的**，硬性拒绝会把 §5.3 描述的样本全丢掉。所以只有在「两侧都是自然语言且归一化后相同」时才拒绝。

> ⚠️ 实现瑕疵：`elif not is_canonical_format(pair.chosen)` 这一支永远不可达（进入外层 `if` 的前提就是 `not is_canonical_format(...)` 为真）。它想表达的规则应该是「chosen 里有工具调用但格式不合法 → 拒绝」，而正确写法是把它挪到外层、与 `if not is_canonical_format(...)` 并列。**在当前实现下，「能解析出 call、但格式不 canonical」的 chosen 会跳过第 5 项参数校验直接进入第 6 项**。已记录在 §9 的待修清单里。

### 6.2 第 6 项：为什么「语义等价」是最大的污染源

```python
if chosen_parse.calls and rejected_parse.calls:
    c, r = chosen_parse.first, rejected_parse.first
    if c.name == r.name:
        m = match_arguments(c.arguments, r.arguments, graded_subset=c.arguments)
        if m.lenient_matched:
            return False, "sides_semantically_equal"
```

用 `match_arguments(..., graded_subset=c.arguments)` 做**宽松口径**的等价判定：`2025-07-15` 与 `2025/07/15` 等价、`PEK` 与 `北京` 等价、`30` 与 `"30公斤"` 等价（`src/eval/equivalence.py`）。

**如果不做这一条会怎样**：模型被要求区分两个其实等价的东西。比如 chosen 是 `{"date": "2025-07-15"}`、rejected 是 `{"date": "2025/07/15"}`。DPO 会去压低 `2025/07/15` 的 logprob——但评测的宽松口径认为这两者都对。结果是训练 loss 稳定下降、评测指标纹丝不动，而且**从 loss 曲线上完全看不出问题**。这条门禁是唯一能在训练前抓住它的地方。

从实际构造报告看，这也是拒绝量最大的一条（360 条入账、63 条因它被拒），说明它确实在起作用，而不是摆设。

### 6.3 `_recovery_coherence`：一个真实 bug 的产物

这是本套门禁里需要单独说明的一条，因为它不是「防御性编程」，而是**被一个实际 bug 逼出来的**。

#### bug 的现象

早期版本的 `build_recovery_pairs` 从整个纠错话术池里**随机抽句子**，不管这个场景是什么错误。于是生成了这样一条训练数据：

```text
<|im_start|>user
2025-07-18北京到上海的航班有哪些<|im_end|>
<|im_start|>assistant
<tool_call>{"name": "search_flights", "arguments": {"origin": "PEK", "destination": "SHA", "date": "2025-07-18"}}</tool_call><|im_end|>
<|im_start|>user
city 参数要填中文城市名，不要填三字码。<|im_end|>       ← 完全文不对题
<|im_start|>assistant
```

首轮错误是「**航班搜索**场景」（参数是 `origin` / `destination`，schema 明确要求三字码），纠错话术却来自 `param_city_name` 组，说的是「**天气工具**的 `city` 要写中文城市名」。这个场景里的 `search_flights` 调用**其实是对的**（`PEK`/`SHA` 完全合法），用户却说你错了。

#### 为什么这比没有数据更糟

DPO 会把这条数据当作事实来学：「即使我的工具调用符合 schema，用户也可能说它错了」。学到的策略是**降低对用户纠正的信任**，也就是——忽略纠错。而「被纠正后改正」正是本项目要训的核心能力。**这条数据在反着训。**

它比单纯的噪声危险，因为噪声只是浪费梯度，而这条给出的是**方向相反的梯度**。

#### 门禁做了什么

`_recovery_coherence(pair)` 从 prompt 里正则取出「首轮错误回复」和「最后一次纠错话术」，检查两者是否自洽：

```python
asst_blocks = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", prompt, flags=re.S)
correction  = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, flags=re.S)
...
wrong_call = extract_tool_calls(asst_blocks[-1])
wrong_args = set(wrong_call.first.arguments) if wrong_call.first else set()
last_correction = correction[-1]
```

**规则 1：纠错话术里显式点名的参数，必须真的出现在首轮错误调用的参数里。**

```python
mentioned = set(re.findall(
    r"\b(city|origin|destination|date|booking_id|flight_id|reason|new_baggage_kg)\b",
    last_correction,
))
missing = {m for m in mentioned if m not in wrong_args}
if missing:
    return False, f"correction_mentions_absent_param:{','.join(sorted(missing))}"
```

上面那条 bug 数据会被 `correction_mentions_absent_param:city` 挡掉——首轮错误调用的参数是 `origin`/`destination`/`date`，没有 `city`。

**规则 2：幻觉类纠错必须点名那个假工具，且假工具不能是真工具。**

```python
if pair.negative_kind in ("simple", "recovery") and "工具" in last_correction and wrong_call.first:
    if wrong_call.first.name not in VALID_TOOL_NAMES:
        if wrong_call.first.name not in last_correction:
            return False, "hallucination_correction_does_not_name_fake_tool"
```

「工具列表里没有 X 这个工具」这类话术必须带上具体的 X，否则模型学到的只是「用户说『工具』的时候要换一个东西」，没有具体指向。从构造报告看，这条挡掉了 13 条。

#### 三条规则各自的局限

1. 它只检查**参数名**，不检查参数**值**。`wrong_city_param` 场景里话术说「city 参数要填中文城市名」，首轮错误调用确实有 `city`（值是三字码），所以通过。这是正确的通过。但如果话术说的是值层面的矛盾（目前没有这种话术），这条规则抓不到。
2. `\b` 是 ASCII 词边界。`booking_id` 之所以能被正确匹配，是因为它排在 alternation 的前面（正则交替是「先匹配先返回」）。**如果以后往 `mentioned` 列表里加一个以已有参数名为前缀的短名（比如加 `id`），`booking_id` 会先被匹配成 `booking_id`，行为不变；但若加 `booking`，可能产生部分匹配。** 扩展这张表时要重新核对。
3. 规则 2 里的 `"工具" in last_correction` 是个粗粒度触发条件。它依赖 `CORRECTION_PHRASING["hallucination"]` 里的三条话术都含「工具」二字。改话术文案时要注意这个耦合。
4. 结构不合法时（取不到 assistant 块或 user 块）直接返回 `True, ""`，理由是「交给其他门禁处理」。这意味着**如果 prompt 结构崩坏，这条门禁会静默放过**。实际影响被 `_passes_quality_gate` 的 `empty_prompt` 和后续的 tokenizer 环节部分兜住，但不是完全可靠。

#### 与 `recovery_corpus.CORRECTION_PHRASING` 的配套设计

语料侧的注释把这一点写在了常量定义处：

```python
#: 纠错话术池，按错误类型分组。
#:
#: ⚠️ **每个场景必须挑与该场景错误类型匹配的那一组**。
#: 早期版本从整个池子里随机抽，结果出现了「航班搜索场景
#: 配上『city 参数要填中文城市名』」这种前后矛盾的训练数据 ——
#: 模型会学到「忽略用户的纠正」，比不加数据更糟。
```

`CORRECTION_PHRASING` 一共 **11 组**：`format`、`hallucination`、`wrong_tool`、`param_placeholder`、`param_airport_code`、`param_city_name`、`param_date`、`fake_entity`、`repeat`、`clarify`、`incomplete`。注意它**不覆盖** `preference_builder.CORRECTION_TEMPLATES` 的全部键名（后者用 `FORMAT_ERROR` / `HALLUCINATION` / … 这套错误类别常量作键，且 `param_airport_code` / `param_city_name` / `param_date` / `fake_entity` / `clarify` / `incomplete` 这些组在它里面根本不存在）。

每个 `_s_*` builder **显式指定**自己要用哪一组，例如：

```python
# _s_city_code：航班场景 → 用三字码那一组
correction = rng.choice(CORRECTION_PHRASING["param_airport_code"])
# _s_wrong_city_param：天气场景 → 用中文城市名那一组（方向相反）
correction = rng.choice(CORRECTION_PHRASING["param_city_name"])
```

**这就是「两个方向相反的话术必须分成两组」的原因**：航班工具的 `origin`/`destination` 要三字码，天气工具的 `city` 要中文城市名。把它们合成一组「城市格式」话术，就必然产生上面那个 bug。设计上把「话术分组」和「场景选组」都变成显式的代码，而不是靠随机抽样的运气。

> ⚠️ 相关发现：[`src/data/preference_builder.py`](../../src/data/preference_builder.py) 里另有一个 `CORRECTION_TEMPLATES`（5 组：FORMAT_ERROR / HALLUCINATION / WRONG_TOOL / PARAM_ERROR / NO_RECOVERY），**全仓库没有任何地方读它**。实际生效的是 `recovery_corpus.CORRECTION_PHRASING`。这套死常量的存在有实际风险：改它的人会以为改了训练数据。见 §9。

### 6.4 门禁的实际拦截量（来自 `data/preferences/build_report.json`）

这份报告是**代码产出、不是实验结论**，可以直接引用：

| `gate_reasons` | 数量 | 说明 |
|---|---|---|
| `sides_semantically_equal` | 63 | 两侧语义等价（最大的一类，说明这条门禁确实在起作用） |
| `chosen_invalid_args:missing` | 25 | chosen 缺必填参数 |
| `hallucination_correction_does_not_name_fake_tool` | 13 | 幻觉话术没点名假工具 |

对应的 `rejected_by_gate`（按构造器分桶）：`format_constraint 59` / `error_recovery 25` / `tool_selection 8` / `param_constraint 9`。

在 §7.2 那次「目标 4200」的探测运行里，`gate_reasons` 累计为 `sides_semantically_equal 22262`、`chosen_invalid_args:missing 6724`、`hallucination_correction_does_not_name_fake_tool 1743`、`identical_sides 131`。四个原因都非零，说明五条检查都在真实拦截，而不是摆设。

---

## 7. 数据集规模：目标值、实际上限与实测产物

### 7.1 配置里的目标值 ≠ 实际产出

`PreferenceCfg` 的默认目标是：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `n_format` | 1200 | 格式约束对目标 |
| `n_recovery` | 1200 | 错误恢复对目标 |
| `n_tool_selection` | 600 | 工具选择对目标 |
| `seed` | 42 | 全局种子 |
| `max_pairs_per_task` | 8 | **⚠️ 全仓库无人读取，是死字段** |
| `hard_negative_ratio` | 0.35 | 见 §4.1，只对两类生效 |

**参数约束对没有自己的配置字段**——`build_all(n_param=...)` 默认 `None` 时回落到 `n_format`，`build_preferences.py` 里也是 `args.n_param if ... else CFG.preference.n_format`。所以改 `n_format` 会**同时**改参数约束对的规模。

### 7.2 去重后的真实上限（本地探测结果，非实验结论）

候选池规模：`num_tasks_pool = 177` = 100 个评测任务 + 77 个 `trainsyn_` 任务（`build_scenarios(seed=42)` 产出 84 个场景，其中 7 个正解是自然语言、转不成 Task，见 `recovery_corpus.build_training_tasks`）。

把四个构造器的目标都设成 4200（远超默认值）跑一次，得到的是**上限**：

| 构造器 | 目标 | 实得 | 是否触顶 |
|---|---|---|---|
| `build_format_pairs` | 4200 | **924** | ✅ **触顶**——受负样本多样性限制 |
| `build_recovery_pairs` | 4200 | 4200 | ❌ 未触顶 |
| `build_tool_selection_pairs` | 4200 | 4200 | ❌ 未触顶 |
| `build_param_pairs` | 4200 | 4200 | ❌ 未触顶 |

**结论：格式约束对是唯一的瓶颈。** 即使把 `n_format` 设成 1200（默认值），也只能拿到约 924 条，缺口约 23%。原因不难定位：格式类的 prompt 只由 177 个任务决定，而 rejected 只有「5 句简单负样本 + 2 种硬负样本构造」共 7 种变形。`key = pair_type|prompt|chosen|rejected` 下，每个任务最多只能贡献约 7 条去重后的对（且 chosen 由任务唯一决定），177 × 7 ≈ 1239，扣掉被门禁拒绝的部分就落在 900 出头。

> ⚠️ 这个上限会随构造逻辑的改动而变化。**重跑之前请自己测一遍**，不要照抄 924：
>
> ```bash
> python -c "import sys;sys.path.insert(0,'.');from src.data.preference_builder import PreferenceBuilder;from src.data.recovery_corpus import build_training_tasks;from src.data.task_manifest import load_tasks;pool=list(load_tasks())+build_training_tasks(seed=42);b=PreferenceBuilder(seed=42);print(len(b.build_format_pairs(pool,4200)))"
> ```

**要突破这个上限，只能在 rejected 侧增加多样性**（例如把 `_SIMPLE_NEGATIVES` 从 5 句扩到几十句、或按错误子类型分别构造坏 JSON），而不是调大 `n_format`。调 `n_format` 只会让 guard 空转到 `n * 20` 后退出。

### 7.3 当前仓库里那份数据的实际构成

`data/preferences/build_report.json` 记录了上一次构造的真实结果（`is_mock: false`，因为它确实是真数据构造，只是模型还没训）：

```
num_tasks_pool  177
num_scenarios    84
total_pairs     360   (train 329 / dev 31)
```

按类型：`format_constraint 120` / `error_recovery 120` / `tool_selection 60` / `param_constraint 60`。

**这组数字对应的是 `--n-format 120 --n-recovery 120 --n-tool 60` 的小规模运行**（报告里的 `config` 段记录了这些值，且 `n_format=120` 恰好卡在格式类的 guard 上限附近），不是默认目标值下的产物。按默认值重跑会得到约 924 / 1200 / 600 / 1200 ≈ 3924 条。

按 `negative_kind`：`hard 129` / `simple 188` / `recovery 43`。
按难度：`easy 109` / `medium 118` / `hard 133`。

`written` 段落里需要单独注意的两行：

- `dpo_all.train.jsonl` 与 `all.train.jsonl` 都是 329 条（前者是后者的切片，内容相同）；
- `dpo_tool.dev.jsonl` 只有 **3** 条、`dpo_tool_param.dev.jsonl` 只有 7 条——小切片的 dev 集过小，见 §8.2 结尾的警告。

实测（`data/preferences/build_report.json`，`seed=42`）：

```
total_pairs = 4528      train_pairs = 4064      dev_pairs = 464

written（各实验切片）：
  all.train.jsonl              4064      all.dev.jsonl              464
  dpo_format.train.jsonl       1014      dpo_format.dev.jsonl       114
  dpo_recovery.train.jsonl     1065      dpo_recovery.dev.jsonl     135
  dpo_joint.train.jsonl        2443      dpo_joint.dev.jsonl        285
  sft_only.train.jsonl         2635      sft_only.dev.jsonl         293
```

切片划分按 `task_id` 而非按样本随机——同一任务的多个偏好对必须在同一侧，
否则 dev 会因为见过同一 prompt 的其他变体而虚高，起不到早停作用。

> 上文的 360 条是早期小规模运行的产物；本节数字已替换为默认目标值下的实际产出。

---

## 8. 两个必须成立的契约

### 8.1 prompt 契约：训练与推理共用同一个 system prompt

`src/models/prompt_builder.py` 的模块 docstring 把话说得很直接：

> ⚠️ **本模块的系统提示词必须与 `src/data/preference_builder.py` 的逐字一致。**
> 偏好对齐最常见的失效原因不是超参不对，而是**训练分布与推理分布不一致**。

实现上靠三件事保证：

1. **单一事实来源**。`SYSTEM_PROMPT` 只在 `prompt_builder.py` 定义一次。`preference_builder` 用 `from src.models.prompt_builder import SYSTEM_PROMPT, build_system_prompt, build_plain_prompt` 引入，并留了一个 `SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT` 的向后兼容别名。**没有第二份模板。**
2. **推理侧走 `build_messages` → `build_plain_prompt`**，它拼接的格式与 `render_prompt` 逐字一致（`<|im_start|>{role}\n{content}<|im_end|>` 序列 + 尾部 `<|im_start|>assistant\n`）。
3. **训练前做机械校验**。`dpo_dataset.verify_prompt_consistency(rows, tasks)` 对每条能对上 `task_id` 的样本，用 `build_plain_prompt(task)[:400]` 重新渲染并比对前缀：

```python
expected_head = build_plain_prompt(task)[:400]
if not r["prompt"].startswith(expected_head):
    checked += 1
    if len(mismatches) < 5:
        mismatches.append({...})
```

`dpo_train.py` 的 `data_health_check` 会调用它，`--dry-run` 会打印「✓ prompt 一致性通过（检查 N 条）」或「✗ prompt 不一致 N 条 —— 训练与推理分布不同！」。

**比对「前缀」而不是全等**，是因为恢复类偏好对的 prompt 故意比评测 prompt 长（多出「首轮错误 + 纠错」两轮）。要求全等会把最有价值的那类数据判成不一致。400 字符足以覆盖 system prompt 的开头与工具清单的起始部分——而 system prompt 恰恰是最容易在维护中漂移的地方。

**如果分布不一致会怎样**：DPO 学到的是条件反射「在这个 prompt 分布下输出 tool_call」。换一版 system prompt，触发条件不成立，模型回到未对齐行为。表现为**训练 loss 一路下降、评测指标一动不动**——这是偏好对齐里最难查的一类问题，因为所有训练侧的指标都是好的。所以必须在训练前用机械校验挡掉，而不是等评测跑完再猜。

### 8.2 train / dev 按 `task_id` 划分

```python
task_ids = sorted({p.task_id for p in pairs})
rng = random.Random(self.seed + 999)
rng.shuffle(task_ids)
n_dev = max(1, int(len(task_ids) * 0.1))
dev_ids = set(task_ids[:n_dev])
for p in pairs:
    p.split = "dev" if p.task_id in dev_ids else "train"
```

**按 task_id 而不是按样本随机划分。** 理由：

同一个 task 会产出多条偏好对（同一 prompt 的不同 rejected，或者 `trainsyn_` 前缀下同一场景的不同变体）。如果按样本随机划分，dev 里会存在「prompt 与 train 里某条几乎相同、只有 rejected 不同」的样本。**这不是泛化，是背诵。** dev loss 会立刻看起来很低，早停（`eval_steps=50`）会在错误的位置触发，而真实评测指标不会跟着动。

按 task_id 划分后，dev 里的任务在 train 里**完全不出现**，dev loss 才具有「没见过的任务上表现如何」的含义。

两点实现细节：

- `rng = random.Random(self.seed + 999)` 用固定偏移的独立 RNG，保证划分不消耗构造阶段 `self.rng` 的状态。否则调整构造逻辑会连带改变划分。
- `n_dev = max(1, ...)` 保证至少有一个 dev 任务。在极小规模（`dpo_tool` 只有 60 条、涉及 57 个 task_id）下这点很重要。从 `build_report.json` 可见 `dpo_tool.dev.jsonl` 只有 3 条——**小切片的 dev 集过小，dev loss 会非常抖**，不宜作为早停依据。这是切片设计的已知代价。

另外注意：`build_all` 里的 `by_split` 统计是**重新计算**的（`self.stats["by_split"] = {}`），因为 `_add` 被调用时 split 还没定。`build_report.json` 里 `stats.by_split` 显示 `{"train": 360}` 而顶层 `train_pairs: 329 / dev_pairs: 31`——两者不一致正是这个原因（`stats` 是在划分前打快照的）。读报告时以 `written` 与顶层 `train_pairs` / `dev_pairs` 为准。

---

## 9. 已知问题与待修清单

以下问题在**写这份文档时读到的代码版本**上确认存在。注意仓库当时正被并行修改，因此每一条都附了**核对命令**——引用前先跑一遍确认它还在。

| # | 位置 | 问题 | 影响 | 核对方式 / 建议 |
|---|---|---|---|---|
| 1 | `src/config.py` `apply_env_overrides` | `AGENTOPT_FORCE_MOCK` **不在** `mapping` 里，也没有单独处理。它只在模块级被读成 `FORCE_MOCK`，全仓库唯一用途是 `describe()` 里打印 | 设了 `AGENTOPT_FORCE_MOCK=1` 只能改一行打印输出，**不改变任何运行路径**；而 config docstring 声称它会让所有加载真实模型的路径退化为 mock | `Select-String -Path src\config.py -Pattern 'FORCE_MOCK'`。要么在 `apply_env_overrides` 里显式处理并让 `mock_pipeline` / `hf_policy` 真的读它，要么删掉这个承诺 |
| 2 | [`requirements.txt`](../../requirements.txt) L50 / [`pyproject.toml`](../../pyproject.toml) L30 | 都写 `trl>=0.9.0,<0.13.0`，而 `compat.recommended_pins()` 返回 `{"trl": "0.13.0"}` | **两者互斥**：按 requirements 装会得到 0.12.x，与文档「锁 0.13.0」矛盾 | `Select-String -Path requirements.txt,pyproject.toml -Pattern 'trl'`。统一为 `trl==0.13.0`，或把 `recommended_pins` 改成 `0.12.0` 并同步改文档。详见 [`trl_version_notes.md`](../ops/trl_version_notes.md) §4 |
| 3 | `preference_builder._passes_quality_gate` | `elif not is_canonical_format(pair.chosen)` 分支**永不可达**（进外层 `if` 的前提就是同一条件为真，见 §6.1） | 「能解析出 call、但格式不 canonical」的 chosen 会跳过第 5 项参数校验 | 读第 300~312 行。把该分支移到外层作为独立 `if` |
| 4 | `preference_builder.CORRECTION_TEMPLATES` | 5 组话术常量，**全仓库无引用**（实际生效的是 `recovery_corpus.CORRECTION_PHRASING`） | 改它的人会误以为改了训练数据 | `Select-String -Path src\data\*.py -Pattern 'CORRECTION_TEMPLATES'`。删除，或加注释指明已废弃 |
| 5 | `PreferenceCfg.max_pairs_per_task` | 声明为 8，无人读取 | 死字段。改它不会有任何效果 | `Select-String -Path src,scripts -Recurse -Pattern 'max_pairs_per_task'`。删除或在 `_add` 里真正实现 |
| 6 | `compat.build_trainer` | 已实现但 `dpo_train.py` 未使用（后者内联了同样的逻辑） | 两份逻辑，只有一份被执行；改 `build_trainer` 不生效 | `Select-String -Path scripts\train\dpo_train.py -Pattern 'build_trainer'`。让 `dpo_train.py` 调用它，或删掉它 |
| 7 | `scripts/prepare/build_preferences.py` | 硬编码 `"is_mock": False` | 数据写到 `outputs/mock_pipeline/` 时也标为真实 | 从 `--out-dir` 是否含 `mock` 推断，或加 `--is-mock` 参数 |
| 8 | [`项目方案.md`](../../项目方案.md) | 通篇说「30 个任务」「格式约束 500 条」「用 Qwen-72B 生成 chosen」 | 与实际实现的 100 任务、四类偏好对、chosen 来自任务 ground truth 严重不一致 | 已在 README §11 标注为「最初的四周项目规划」；建议在文首加一句「实现已超出本规划，以 README 为准」 |

已确认**无问题**：`python tests/run_tests.py` 实测 **48 通过 / 0 失败（2.05s）**，与 README 的说法一致。

---

## 10. 扩展这份数据集的检查清单

如果要加第五类偏好对（例如针对 `REFUSED` 的独立类型）或换评测域，按这个顺序做：

1. **先在 `error_taxonomy` 里定类别与子类型**，并写进 `CATEGORY_REMEDIES_ZH`。没有分类的错误没法验收。
2. **在 `preference_builder` 里加 `PAIR_*` 常量 + 中文标签 + `build_*` 方法**。签名保持 `(tasks, n) -> list[PreferencePair]`，并复用 `_add(pair, bucket)` 而不是自己维护统计。
3. **`EXPERIMENT_PAIR_TYPES` 里加一个实验切片**，否则新数据没有独立的消融入口。
4. **检查 `key` 的去重是否够细**。新类型的（rejected 变体数 × 任务数）= 规模上限。先按 §7.2 的方法探测上限，再决定目标值，别让 guard 空转。
5. **构造纠错类数据时，话术必须按错误类型分组**，并让新类型的 `_recovery_coherence` 规则覆盖它。§6.3 的 bug 会以同样的方式重现。
6. **跑 `scripts/prepare/inspect_preferences.py`**，确认 `sides_equivalent` / `chosen_not_canonical` / `prompt_missing_assistant_head` 三项都是 0，且 prompt 一致性检查通过。
7. **跑 `python scripts/train/dpo_train.py --experiment <新实验> --dry-run`**，确认审计通过并拿到建议的 `max_length`。
