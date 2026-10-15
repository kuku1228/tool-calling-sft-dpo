"""「已知限制」自动生成模块的测试。

## 为什么这个也要测

这个项目里最危险的错误不是「算错」，而是「**报告漏掉了某个限制**」——
数字对，但结论被读成更强的说法。实测发生过三处：

1. `seen`/`unseen` 被当成泛化差距（那个标签**从不读取训练数据**）
2. 「14 个 clarify 独立样本」（有效样本是 1 个行为）
3. 剔除泄漏后样本量变了，报告标题没跟着变

所以我把它做成自动章节。**但如果这个模块自己坏了（例如永久返回空列表），
报告就会静默地失去限制说明** —— 那和没有它一样危险。

所以测试要覆盖两个方向：
* 有真限制时**必须报出来**
* 每条限制必须**明确写出「不能怎么声称」**（这是最有用的字段）
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.eval.limitations import collect_limitations, render_markdown  # noqa: E402
from tests._harness import A, test  # noqa: E402


@test
def test_collect_limitations_finds_the_known_ones() -> None:
    """当前配置下必须报出至少 3 条限制（都是实测确认过的）。

    没有这条测试的话，模块可能因为某个 ``except`` 吞掉异常而**返回空列表**，
    报告于是静默失去限制说明 —— 而它看起来完全正常。
    """
    from src.data.task_manifest_v2 import load_tasks_v3

    lim = collect_limitations(load_tasks_v3(), task_set="v3")
    ids = {x["id"] for x in lim}
    A.true(len(lim) >= 3, f"应报出至少 3 条限制，实际 {len(lim)}：{sorted(ids)}")
    # 这两条是实测确认的硬事实，必须报出来
    A.true("visibility_meaningless" in ids,
           f"必须报出 seen/unseen 无含义，实际 {sorted(ids)}")
    A.true("duplicate_expected_calls" in ids,
           f"必须报出重复期望调用，实际 {sorted(ids)}")


@test
def test_every_limitation_states_what_not_to_claim() -> None:
    """每条限制都必须有 ``do_not_claim`` —— 否则它只是「有个限制」的废话。

    最有用的信息不是「存在限制」，而是「**所以哪个说法不能用**」。
    实测：看见「seen/unseen 有局限」不会让人停止讲泛化，
    但看见「不要把差值解释为泛化差距」会。
    """
    from src.data.task_manifest_v2 import load_tasks_v3

    lim = collect_limitations(load_tasks_v3(), task_set="v3")
    for x in lim:
        A.true(bool(x.get("do_not_claim")),
               f"{x['id']} 缺少 do_not_claim —— 读者不知道该避免什么说法")
        A.true(len(x["do_not_claim"]) > 20,
               f"{x['id']} 的 do_not_claim 过短，说不清问题")
        A.true(bool(x.get("statement")), f"{x['id']} 缺少 statement")
        A.true(bool(x.get("severity")), f"{x['id']} 缺少 severity")


@test
def test_duplicate_call_limitation_reports_the_real_cluster_size() -> None:
    """重复期望调用那条必须报**真实簇大小**，不是硬编码的 15。

    数据改了（例如将来把 clarify 扩成多种缺失模式），
    簇大小会变，文字必须跟着变 —— 否则报告会引用过期的数字。
    """
    from src.data.task_manifest_v2 import load_tasks_v3

    lim = collect_limitations(load_tasks_v3(), task_set="v3")
    d = next(x for x in lim if x["id"] == "duplicate_expected_calls")
    # 当前实测：106 任务 → 92 个不同期望调用，最大簇 15
    A.true("92" in d["statement"],
           f"应报出真实的簇数 92，实际：{d['statement'][:80]}")
    A.true("106" in d["statement"], "应报出任务数 106")
    A.true("1 个行为" in d["do_not_claim"],
           "必须点明「有效独立样本是 1 个行为」")


@test
def test_render_markdown_includes_all_limitations_and_headers() -> None:
    """渲染必须包含每条限制，且带「不能这样声称」的标题。"""
    from src.data.task_manifest_v2 import load_tasks_v3

    tasks = load_tasks_v3()
    lim = collect_limitations(tasks, task_set="v3")
    md = render_markdown(lim, n_tasks=len(tasks), label="测试")
    A.true("## 已知限制" in md, "应有章节标题")
    A.true("自动生成" in md, "应声明是自动生成的（读者才知道可信度来源）")
    A.true("不能这样声称" in md, "必须有「不能这样声称」的小节")
    for x in lim:
        A.true(x["id"] in md, f"{x['id']} 未出现在渲染结果里")
    A.true("106" in md, "应包含任务数")


@test
def test_render_markdown_handles_empty_limitations() -> None:
    """没有限制时不得崩溃，且要说明「未检出」。"""
    md = render_markdown([], n_tasks=10, label="空")
    A.true("未检出" in md, f"空列表时应说明未检出，实际：{md[:60]}")


@test
def test_limitations_are_written_into_the_analysis_json() -> None:
    """限制必须同时落进 ``error_analysis.json``。

    报告若只引用 json（而不是读 md），也能看到限制 ——
    两条路径都要有，避免「md 写了但 json 里没有」。
    """
    # 用 mock 跑一次真实评测脚本，检查产物
    import subprocess
    import tempfile

    label = "limtest_tmp"
    out = _REPO / "experiments" / label
    if out.exists():
        import shutil

        shutil.rmtree(out, ignore_errors=True)
    try:
        subprocess.run(
            [sys.executable, "-u", "scripts/eval/run_baseline.py",
             "--label", label, "--task-set", "v3", "--mock",
             "--num-samples", "1", "--task-limit", "20"],
            cwd=str(_REPO), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600,
        )
        md = out / "limitations.md"
        A.true(md.exists(), "应生成 limitations.md")
        import json

        an = json.loads((out / "error_analysis.mock.json").read_text(encoding="utf-8"))
        A.true("limitations" in an, "error_analysis.json 里应含 limitations 字段")
        A.true("n_tasks" in an, "error_analysis.json 里应含 n_tasks（样本量要显式）")
        A.eq(an["n_tasks"], 20, "n_tasks 应反映实际评测任务数")
    finally:
        import shutil

        shutil.rmtree(out, ignore_errors=True)
