"""轻量测试运行器（不依赖 pytest）。

用法
----
::

    python tests/run_tests.py                 # 跑全部
    python tests/run_tests.py --verbose       # 每个测试都打印
    python tests/run_tests.py --filter grader # 只跑名字含 grader 的
    python tests/run_tests.py --list          # 只列出测试名

也兼容 pytest：``python -m pytest tests/ -q``（若已安装）。
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行单元测试")
    p.add_argument("--verbose", "-v", action="store_true", help="打印每个测试")
    p.add_argument("--filter", default=None, help="只跑名字包含该字符串的测试")
    p.add_argument("--list", action="store_true", help="只列出测试名")
    p.add_argument("--stop-on-fail", action="store_true", help="首个失败即停止")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # 导入测试模块以触发注册
    import tests.test_agent  # noqa: F401
    import tests.test_core  # noqa: F401
    import tests.test_methodology  # noqa: F401
    # 任务集质量约束（v2）—— 与评测正确性同等重要：
    # 任务集一倒，上面所有结论都倒，而这些缺陷**不会在评测时报错**。
    import tests.test_task_set_v2  # noqa: F401
    # 配对统计检验 —— 我长期用「CI 是否重叠」判断显著性，那是错的。
    # 换判据同样可能悄悄错，所以把它的前提与退化情形都测出来。
    import tests.test_stats  # noqa: F401
    # 评测集与训练语料的**重合层级**审计。
    # 「只查 prompt 是否逐字相同」会漏掉「期望调用相同」这类泄漏 —— 实测漏了 5 个。
    import tests.test_overlap  # noqa: F401
    # 文档里不得**无标注地**引用已作废的数字。
    # 这类过期引用会让文档主动误导读者（面试时会照着讲），而且不会报错。
    import tests.test_docs  # noqa: F401
    # 「已知限制」自动生成 —— 报告漏掉限制比算错数字更危险：
    # 数字对，但结论被读成更强的说法。
    import tests.test_limitations  # noqa: F401
    from tests._harness import _TESTS

    tests = _TESTS
    if args.filter:
        tests = [(n, f) for n, f in tests if args.filter.lower() in n.lower()]

    if args.list:
        for name, _ in tests:
            print(name)
        print(f"\n共 {len(tests)} 个测试")
        return 0

    if not tests:
        print("没有匹配的测试。", file=sys.stderr)
        return 1

    print("=" * 74)
    print(f"运行 {len(tests)} 个测试")
    print("=" * 74)

    passed = 0
    failed: list[tuple[str, str]] = []
    t_start = time.time()

    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            passed += 1
            if args.verbose:
                print(f"  ✓ {name}  ({time.time() - t0:.2f}s)")
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ✗ {name}")
            print(f"      {type(exc).__name__}: {exc}")
            if args.verbose:
                print("      " + "\n      ".join(tb.strip().splitlines()[-8:]))
            if args.stop_on_fail:
                break

    dt = time.time() - t_start
    print("=" * 74)
    print(f"结果: {passed} 通过, {len(failed)} 失败, 用时 {dt:.2f}s")
    if failed:
        print()
        print("失败清单:")
        for name, err in failed:
            print(f"  ✗ {name}")
            print(f"      {err}")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
