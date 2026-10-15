"""文档「作废数字」检查器的测试。

## 为什么这个也要测

这个检查器守的是**文档的可信度**。它自己出错有两个方向，都危险：

* **漏报**：作废数字没被标记出来 → 文档主动误导读者（面试时会照着讲）
* **误报**：把「我推翻了 95% 那个结果」这种**正确引用**也报成违规
  → 检查器被忽略，等于没有

实测两个方向都踩过：
* 漏报：`agent_eval_report.md` 加了文首横幅，但数字在 L111+、
  3 行窗口内没有标记 → 仍报 13 处违规
* 误报：表格写成 `| 模型 | 95.00% | 污染数据 |`，
  「污染数据」就在同一行，早期判据不认，报了 80 处假违规
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests._harness import A, test  # noqa: E402


def _load_module():
    path = _REPO / "scripts/report/check_stale_numbers.py"
    spec = importlib.util.spec_from_file_location("_stale_chk", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@test
def test_checker_flags_an_unannotated_stale_number() -> None:
    """**反向测试**：没标注的作废数字必须被抓到。

    没有这条，前面「108 处全部标注」可能只是因为判据永远是「已标注」。
    """
    m = _load_module()
    ls = ["# 报告", "", "我们的 SFT 达到 95.00% 准确率。"]
    A.false(m.line_has_marker(ls, 2), "无标注的行不应被判为已标注")
    A.false(m.doc_is_deprecated(ls), "文首无声明时不应判为整份作废")


@test
def test_checker_accepts_inline_marker_on_same_line() -> None:
    """表格状态列与数字在同一行时，必须认作已标注。

    实测：``| 模型 | 95.00% | 污染数据 |`` —— 早期判据不认同行，
    报了 80 处假违规，检查器因此形同虚设。
    """
    m = _load_module()
    A.true(m.line_has_marker(["| sft_only | 95.00% | 污染数据 |"], 0))
    A.true(m.line_has_marker(["| sft_v3 | 83.00% | 跑在 v1 上 |"], 0))
    A.true(m.line_has_marker(["95.00%（已作废）"], 0))
    A.true(m.line_has_marker(["95.00%（等 v3 重跑）"], 0))


@test
def test_checker_accepts_nearby_marker() -> None:
    """前后 3 行内有标注也算 —— 段落式说明很常见。"""
    m = _load_module()
    ls = ["此结果不可引用：", "它跑在 v1 上。", "95.00% 是污染版"]
    A.true(m.line_has_marker(ls, 2))


@test
def test_doc_level_deprecation_covers_later_numbers() -> None:
    """文首声明「整份作废」后，后面很远的数字也应视为已标注。

    实测：`agent_eval_report.md` 的横幅在 L3-5、数字在 L111+，
    只靠 3 行窗口会报 13 处违规。**一份文档声明了整体作废，
    就不该逐个数字再标一遍。**
    """
    m = _load_module()
    head = ["# Agent 评测报告", "", "> # ⚠️ 本报告的全部数字已作废"]
    tail = ["x"] * 100 + ["轨迹成功率 100.00%"]
    A.true(m.doc_is_deprecated(head), "应识别出文首的整份作废声明")
    # 后面那行本身没有标记，但文档级声明应覆盖它
    A.true(m.line_has_marker(head + tail, len(head) + 100) is False,
           "该行附近确实没有就近标注（用来证明覆盖来自文档级）")
    A.true(m.doc_is_deprecated(head + tail), "文档级声明对尾部同样有效")


@test
def test_doc_level_markers_cover_unstandard_wording() -> None:
    """不同文档的措辞不同，标记列表必须都覆盖。

    实测漏报：`resume_entries.md` 写「旧版的所有性能数字已作废」、
    `interview_prep.md` 写「本稿的**数字全部作废**」——
    两句话意思相同但措辞不同，早期只认一种，误报 17 处。
    """
    m = _load_module()
    wordings = [
        "> # ⚠️ 状态：本稿的**数字全部作废**",
        "> 旧版的所有性能数字已作废",
        "> ⚠️⚠️ 本报告的全部数字**已作废**",
        "> 本报告的数字存在数据污染，结论需重新验证",
        "> 本稿有两处**判据错误**",
    ]
    for w in wordings:
        A.true(m.doc_is_deprecated(["# 标题", w]),
               f"应识别这种措辞: {w[:40]}")


@test
def test_current_docs_have_no_unannotated_stale_numbers() -> None:
    """当前仓库的文档不得有未标注的作废数字。

    这是**回归测试**：以后改动文档时若引入旧数字而无标注，这里会失败。
    """
    m = _load_module()
    files = sorted(_REPO.glob("docs/**/*.md"))
    A.true(len(files) >= 10, f"应找到多份文档，实际 {len(files)}")
    bad: list[str] = []
    for f in files:
        for h in m.check_file(f):
            if not h["annotated"]:
                bad.append(f"{Path(h['file']).name}:L{h['line']} {h['number']}")
    A.eq(bad, [], f"存在未标注的作废数字：{bad[:6]}")


@test
def test_stale_list_is_not_empty_and_has_reasons() -> None:
    """作废清单必须非空，且每条都要有原因。

    清单是**有意维护的负担**：每次作废一批数字都要在这里登记。
    没有原因的话，看到警告的人不知道该信哪个。
    """
    m = _load_module()
    A.true(len(m.STALE_NUMBERS) >= 10,
           f"作废清单过短，实际 {len(m.STALE_NUMBERS)} 条")
    for pat, why in m.STALE_NUMBERS.items():
        A.true(len(why) > 3, f"{pat} 缺少作废原因")

# ---------------------------------------------------------------------------
# 部署保护逻辑（「保护过度」这类错误只在本地之外暴露）
# ---------------------------------------------------------------------------
def _load_deploy_module():
    import importlib.util

    path = _REPO / "scripts/deploy/update_in_place.py"
    spec = importlib.util.spec_from_file_location("_deploy_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_protected(rel: str) -> bool:
    m = _load_deploy_module()
    protected = any(
        rel == p or rel.startswith(p + "/") for p in m.PROTECTED_PREFIXES
    )
    if protected and Path(rel).name in m.PROTECTED_ALLOW_FILES:
        return False
    return protected


@test
def test_deploy_protects_results_but_updates_data_files() -> None:
    """部署脚本必须**保护实验结果、但放行数据文件**。

    ## 实测踩到的坑

    ``experiments/`` 里**混着两类东西**：实验**结果**（不能覆盖）与
    **数据文件**（必须跟着代码更新）。早期只按目录前缀保护，于是
    ``experiments/_frozen_fingerprint.json`` 被一起跳过 ——
    云端 `test_frozen_eval_tasks_unchanged` 报「缺少冻结指纹文件」，
    **159 个测试挂 1 个**。

    这类「保护过度」的错误**在本地不会暴露**（文件本来就在），
    只在部署后出现。所以必须用路径判定来测。
    """
    A.true(_is_protected("experiments/baseline/eval_report.json"),
           "真实实验结果必须被保护")
    A.true(_is_protected("experiments/sft_v3/eval_report.json"),
           "真实实验结果必须被保护")
    A.true(_is_protected("outputs/sft_v3/final/adapter_model.safetensors"),
           "训练产物必须被保护")

    A.false(_is_protected("experiments/_frozen_fingerprint.json"),
            "冻结指纹是**数据文件**，必须能更新（否则云端测试会挂）")
    A.false(_is_protected("src/data/task_manifest.py"),
            "源码必须能更新")


@test
def test_freeze_script_can_regenerate_the_fingerprint() -> None:
    """指纹文件必须**能被脚本重建** —— 不能只靠一个手工文件。

    原缺口：`test_frozen_eval_tasks_unchanged` 的断言写着
    「应由 scripts 生成」，但**没有那个脚本**。文件一旦丢失就无法重建，
    而错误消息指向一个不存在的能力。

    这条测试同时验证：脚本算出的指纹与现有文件**完全一致**
    —— 也就是 v1 确实从未被改动。
    """
    import importlib.util
    import json

    path = _REPO / "scripts/prepare/freeze_v1_tasks.py"
    spec = importlib.util.spec_from_file_location("_freeze_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    current = mod.compute("v1")
    A.eq(len(current), 100, "v1 应有 100 个任务")

    fp_path = _REPO / "experiments/_frozen_fingerprint.json"
    A.true(fp_path.exists(), "指纹文件应存在")
    frozen = json.loads(fp_path.read_text(encoding="utf-8"))
    diff = mod.compare(current, frozen)
    A.eq(diff["changed"], [], f"v1 被改动了，历史结果不再可比：{diff['changed'][:5]}")
    A.eq(diff["missing"], [], f"v1 任务被删除：{diff['missing'][:5]}")
    A.eq(diff["added"], [], f"v1 任务被新增：{diff['added'][:5]}")

# ---------------------------------------------------------------------------
# 打包流程（我在这里犯过「打完包又改文件」的错）
# ---------------------------------------------------------------------------
def _load_zip_module():
    import importlib.util

    path = _REPO / "scripts/deploy/make_upload_zip.py"
    spec = importlib.util.spec_from_file_location("_zip_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@test
def test_packaging_required_files_all_exist() -> None:
    """``REQUIRED`` 里列的每个文件都必须真的存在于仓库中。

    ## 我在这里栽过一次

    打完包之后**又继续写了几个文件，却没有重新打包**，于是云端部署的是
    中间版本 —— 现象是「云端 163 个测试、本地 170 个」，
    而我一开始还怀疑是部署脚本的问题，查了半天。

    **根因是流程缺陷**：没有任何东西告诉我「这个包是什么时候、包含了什么」。

    这条测试的第一层作用：``REQUIRED`` 里写的路径**不能是错的**
    （写错了会导致打包时报假警报，而假警报会让人忽略这个检查）。
    """
    m = _load_zip_module()
    missing = [r for r in m.REQUIRED if not (_REPO / r).exists()]
    A.eq(missing, [], f"REQUIRED 里列了不存在的文件（路径写错了）：{missing}")


@test
def test_packaging_keeps_the_same_files_as_deploy_protects() -> None:
    """打包的 ``KEEP_IN_EXPERIMENTS`` 必须与部署脚本的
    ``PROTECTED_ALLOW_FILES`` 一致。

    两处不一致的后果很具体：若**包里少了**某个数据文件，而部署脚本
    以为它在，那么云端就会**静默缺文件** —— 这正是
    ``_frozen_fingerprint.json`` 那次的问题（云端测试挂 1 个）。
    """
    zm = _load_zip_module()
    dm = _load_deploy_module()
    A.eq(set(zm.KEEP_IN_EXPERIMENTS), set(dm.PROTECTED_ALLOW_FILES),
         "打包白名单与部署放行白名单必须一致")


@test
def test_packaging_manifest_records_test_count() -> None:
    """打包必须能数出测试数 —— 这是「云端数字对不对」的判据。

    没有它的话，部署后只能靠观察「云端报 163、本地报 170」这种现象
    去猜哪个包被部署了。
    """
    m = _load_zip_module()
    # `count_tests` 会真的跑一次 --list；本地应能数出 >100
    n = m.count_tests()
    A.true(n is not None, "应能从 run_tests.py --list 数出测试数")
    A.true(n and n > 100, f"测试数异常：{n}")

# ---------------------------------------------------------------------------
# 训练监控（判据必须是产物，不是进程）
# ---------------------------------------------------------------------------
def _load_monitor():
    import importlib.util

    path = _REPO / "scripts/train/monitor.py"
    spec = importlib.util.spec_from_file_location("_mon_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@test
def test_monitor_parses_tqdm_progress_lines() -> None:
    """监控必须能从日志里解析出进度（否则只能干等）。"""
    import os

    from tests._harness import workspace_tempdir

    m = _load_monitor()
    d = workspace_tempdir("mon")
    try:
        log = d / "t.log"
        log.write_text(
            "  输入偏好对 2688 条 → SFT 样本 2688 条\n"
            "  截断率=0.0%（max_length=1280）\n"
            "  10/336 [00:52<28:00, 0.19 it/s]\n"
            "  120/336 [10:52<19:24, 0.18 it/s]\n",
            encoding="utf-8",
        )
        prog = m.last_progress(log)
        A.true(prog is not None, "应解析出进度")
        A.eq(prog[0], 120, "应取**最后**一行进度")
        A.eq(prog[1], 336)
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)


@test
def test_monitor_header_report_extracts_truncation_rate() -> None:
    """监控必须报出**截断率** —— 它是最容易被忽略的静默缺陷。

    实测踩过 23.2% 的 prompt 被静默截断，而 loss 曲线和进度条**全都正常**。
    所以这个数字必须在监控里显式出现。
    """
    import shutil

    from tests._harness import workspace_tempdir

    m = _load_monitor()
    d = workspace_tempdir("mon2")
    try:
        log = d / "t.log"
        log.write_text(
            "  输入偏好对 2688 条 → SFT 样本 2688 条\n"
            "  长度: prompt p99=1086  截断率=0.0%（max_length=1280）\n",
            encoding="utf-8",
        )
        rep = "\n".join(m.header_report(log))
        A.true("截断率" in rep, f"应报出截断率，实际：{rep}")
        A.true("0.0%" in rep, f"应报出截断率的数值，实际：{rep}")
        A.true("训练样本" in rep or "SFT 样本" in rep, "应报出样本数")
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test
def test_monitor_judges_by_artifact_not_process() -> None:
    """监控的完成判据必须是**产物 ``final/``**，不是进程存活。

    实测踩过：进程消失后以为完成，实际是 OOM 崩溃。
    进程「正常退出」与「崩溃退出」在 ``pgrep`` 眼里是一样的。
    """
    import importlib.util

    path = _REPO / "scripts/train/monitor.py"
    src = path.read_text(encoding="utf-8")
    A.true("final" in src and "final.exists()" in src,
           "必须用 final/ 是否存在作为完成判据")
    A.true("异常结束" in src, "进程退出但无产物时必须报「异常结束」")

# ---------------------------------------------------------------------------
# SFT 训练的数据文件解析（这里栽过「静默读错文件」）
# ---------------------------------------------------------------------------
@test
def test_sft_experiment_uses_its_own_data_file() -> None:
    """``--experiment X`` 必须用 ``X.train.jsonl``，不能静默读对照组的。

    ## 实测踩到的坑

    ``resolve_data_file`` 原来**先看 yaml 里声明的** ``train_file``，
    而 ``sft_only.yaml`` 写死了 ``sft_only.train.jsonl`` ——
    于是 ``--experiment sft_clarify`` 会读到**对照组**的数据。

    症状：dry-run 报「输入偏好对 **2635** 条」，而它自己的数据是 **2999** 条。
    若没发现，就是**白跑 30 分钟 GPU**，且结果与对照组一模一样，
    看起来像「实验失败」。

    这类错误**不报错**，只让实验白跑 —— 所以必须用测试守住。
    """
    import json

    from src.data.preference_builder import EXPERIMENT_PAIR_TYPES as E

    # 数据文件必须与实验名对应
    for exp in ("sft_only", "sft_clarify"):
        f = _REPO / "data/preferences" / f"{exp}.train.jsonl"
        A.true(f.exists(), f"{exp} 的数据文件应存在：{f}")
        rows = [json.loads(ln) for ln in f.open(encoding="utf-8") if ln.strip()]
        types = {r.get("pair_type") for r in rows}
        # 数据里的 pair_type 应恰好覆盖该实验声明的组合
        declared = set(E[exp])
        A.eq(types & declared, declared,
             f"{exp} 的数据缺少声明的 pair_type：{declared - types}")


@test
def test_sft_experiment_list_includes_clarify_variant() -> None:
    """新加的 SFT 变体必须同时加进 ``SFT_EXPERIMENTS``。

    实测踩过：把 ``sft_clarify`` 加到 ``EXPERIMENT_PAIR_TYPES`` 却忘了
    ``SFT_EXPERIMENTS`` —— dry-run 通过，**真正训练时才报「未知实验」**。
    两处列表不一致就等于「数据生成了但训不了」。
    """
    import importlib.util
    import re

    src_train = (_REPO / "scripts/train/sft_train.py").read_text(encoding="utf-8")
    src_eval = (_REPO / "scripts/eval/run_experiment.py").read_text(encoding="utf-8")

    def _list_of(src: str) -> set:
        m = re.search(r"SFT_EXPERIMENTS[^=]*=\s*\(([^)]*)\)", src)
        return set(re.findall(r'"([^"]+)"', m.group(1))) if m else set()

    a, b = _list_of(src_train), _list_of(src_eval)
    A.true(bool(a), "应从 sft_train.py 解析出列表")
    A.eq(a, b, "两处的 SFT_EXPERIMENTS 必须一致")
    A.true("sft_clarify" in a, f"sft_clarify 应在列表里，实际 {sorted(a)}")

    # 每个 SFT 实验都必须**真的能跑**：要么有数据文件，要么有 pair_type 定义。
    # ⚠️ 实测踩过 `sft_all` 是孤儿（在允许列表里但两样都没有），
    # 于是 `--experiment sft_all` 会**静默退回 yaml 声明的文件**，
    # 读到别的实验的数据。所以这条必须查「能不能真的跑」。
    from src.data.preference_builder import EXPERIMENT_PAIR_TYPES as E

    for exp in sorted(a):
        has_data = (_REPO / f"data/preferences/{exp}.train.jsonl").exists()
        has_decl = exp in E
        A.true(has_data or has_decl,
               f"{exp} 在 SFT_EXPERIMENTS 里，但既没有数据文件、"
               f"也没有 pair_type 定义 —— 会静默读到别的实验的数据")

# ---------------------------------------------------------------------------
# 报告数字必须与实验产物一致（防止报告与数据脱节）
# ---------------------------------------------------------------------------
#: 报告必须覆盖的实验目录
REPORT_EXPERIMENTS: tuple[str, ...] = (
    "baseline", "sft", "dpo_format", "dpo_recovery", "dpo_joint",
)


def _results_text() -> str:
    return (_REPO / "docs" / "ablation" / "results.md").read_text(encoding="utf-8")


@test
def test_results_report_exists_and_covers_all_experiments() -> None:
    """主结果报告必须存在，且覆盖全部实验。"""
    p = _REPO / "docs" / "ablation" / "results.md"
    A.true(p.is_file(), f"找不到主报告 {p}")

    rep = _results_text()
    missing = [e for e in REPORT_EXPERIMENTS if e not in rep]
    A.eq(missing, [], f"报告未覆盖这些实验：{missing}")

    # 报告的标题与出处要写清楚
    A.true("eval_report.json" in rep, "报告应注明数据来源是 eval_report.json")


@test
def test_results_report_numbers_match_the_artifacts() -> None:
    """报告里的准确率/pass^4 必须与实验产物一致。

    报告过期的危害很具体：读者会引用一个与数据不符的数字，而没有任何东西会报错。
    这里把「能从产物直接读出」的数字全部钉住。

    （配对检验的数值无法在本地重算——那需要 GPU。这里只钉可从 JSON 读出的部分。）
    """
    import json

    rep = _results_text()
    missing: list[str] = []
    for exp in REPORT_EXPERIMENTS:
        p = _REPO / "experiments" / exp / "eval_report.json"
        if not p.is_file():
            missing.append(f"{exp}(无 eval_report.json)")
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        h = r["headline"]

        # 准确率与 pass^4 的百分数形式必须能在报告里找到
        for key, val in (("accuracy", h["accuracy"]), ("pass^4", h["pass_hat_k"])):
            pct = f"{val * 100:.1f}%"
            if pct not in rep:
                missing.append(f"{exp}.{key}={pct}")

    A.eq(missing, [], f"报告缺少或与产物不符的数字：{missing[:8]}")


@test
def test_results_report_declares_the_known_limitations() -> None:
    """主报告必须声明三条已知限制。

    漏掉限制比算错数字更危险：数字对，但结论被读成更强的说法。
    """
    rep = _results_text()
    for kw, why in (
        ("seen", "seen/unseen 标签无含义必须声明"),
        ("簇", "有效独立样本少于表面样本必须声明"),
    ):
        A.true(kw in rep, f"报告缺少声明：{why}")


@test
def test_cluster_count_in_report_matches_the_task_set() -> None:
    """报告里「有效独立样本」的簇数必须与任务集实际一致。"""
    from src.eval.paired_stats import expected_keys
    from src.data.task_manifest_v2 import load_tasks_v3

    n_tasks = len(load_tasks_v3())
    n_clusters = len(set(expected_keys("v3").values()))

    rep = _results_text()
    A.true(
        str(n_clusters) in rep,
        f"报告应含有效簇数 {n_clusters}（{n_tasks} 个任务收缩而来）",
    )
    A.true(n_clusters < n_tasks, "本任务集应存在重复的期望调用（簇数 < 任务数）")
