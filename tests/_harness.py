"""单元测试：任务集、工具目录、评分器、指标、损失、偏好数据。

用法
----
::

    python tests/run_tests.py              # 不需要 pytest
    python tests/run_tests.py --verbose
    python -m pytest tests/ -q             # 装了 pytest 也可以

为什么用自带的轻量 runner
------------------------
本项目的设计目标之一是「在一台只装了 numpy 的干净机器上也能验证」。
pytest 不是必需的依赖，所以这里提供一个只依赖标准库的运行器。
它同时兼容 pytest（``test_*`` 函数风格）。
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ============================================================================
# 测试注册机制
# ============================================================================

_TESTS: list[tuple[str, Callable[[], None]]] = []


def test(fn: Callable[[], None]) -> Callable[[], None]:
    """装饰器：注册一个测试函数。"""
    _TESTS.append((fn.__name__, fn))
    return fn


class Assertion:
    """极简断言辅助（比裸 assert 给出更好的错误信息）。"""

    @staticmethod
    def true(cond: Any, msg: str = "") -> None:
        if not cond:
            raise AssertionError(msg or "期望为真，实际为假")

    @staticmethod
    def false(cond: Any, msg: str = "") -> None:
        if cond:
            raise AssertionError(msg or "期望为假，实际为真")

    @staticmethod
    def eq(a: Any, b: Any, msg: str = "") -> None:
        if a != b:
            raise AssertionError(msg or f"期望 {b!r}，实际 {a!r}")

    @staticmethod
    def close(a: float, b: float, tol: float = 1e-6, msg: str = "") -> None:
        if abs(a - b) > tol:
            raise AssertionError(msg or f"期望 {b}±{tol}，实际 {a}")

    @staticmethod
    def in_(item: Any, container: Any, msg: str = "") -> None:
        if item not in container:
            raise AssertionError(msg or f"{item!r} 不在 {container!r} 中")

    @staticmethod
    def raises(exc_type: type, fn: Callable[[], Any], msg: str = "") -> None:
        try:
            fn()
        except exc_type:
            return
        except Exception as e:  # noqa: BLE001
            raise AssertionError(
                msg or f"期望抛 {exc_type.__name__}，实际抛 {type(e).__name__}: {e}"
            ) from e
        raise AssertionError(msg or f"期望抛 {exc_type.__name__}，但没有抛异常")


A = Assertion


# ============================================================================
# 临时目录
# ============================================================================

#: 测试产物根目录。**故意不用 tempfile** ——
#: 在受限沙箱（例如 DSH 的 workspace-write 模式）下，
#: 系统临时目录是不可写的，用 tempfile 会让测试因为环境问题而假失败。
#: 项目内的 .tmp/ 一定可写（已被 .gitignore 忽略）。
TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp"


def make_temp_dir(prefix: str = "case") -> Path:
    """在仓库内创建一个唯一的临时目录。

    >>> p = make_temp_dir("demo"); p.exists()
    True
    >>> import shutil; shutil.rmtree(p, ignore_errors=True)
    """
    import os
    import uuid

    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    p = TEST_TMP_ROOT / f"{prefix}_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p


class TempDir:
    """上下文管理器：退出时清理。

    >>> with TempDir("ctx") as d:
    ...     _ = (d / "f.txt").write_text("x", encoding="utf-8")
    ...     print((d / "f.txt").exists())
    True
    """

    def __init__(self, prefix: str = "case", cleanup: bool = True) -> None:
        self.prefix = prefix
        self.cleanup = cleanup
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.path = make_temp_dir(self.prefix)
        return self.path

    def __exit__(self, *exc: Any) -> None:
        import shutil

        if self.cleanup and self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)

def workspace_tempdir(prefix: str = "t") -> "Path":
    """在工作区里造一个临时目录并返回。

    ⚠️ 为什么不用 ``tempfile``：在 Windows 沙箱下系统 temp
    （``C:\\WINDOWS\\TEMP``）**连写入都拒绝**
    （``PermissionError: [WinError 5] 拒绝访问``）。
    实测两个纯统计逻辑的测试因此失败 —— 而失败原因与它们要测的东西
    毫无关系。工作区是可写的，所以临时目录放在这里。

    调用方负责清理；用 ``shutil.rmtree(root, ignore_errors=True)``。
    """
    import itertools
    import os
    from pathlib import Path as _P

    root = _P(__file__).resolve().parents[1] / ".tmp"
    root.mkdir(parents=True, exist_ok=True)
    for i in itertools.count():
        d = root / f"{prefix}_{os.getpid()}_{i}"
        try:
            d.mkdir()
            return d
        except FileExistsError:
            continue
    raise RuntimeError("无法创建临时目录")
