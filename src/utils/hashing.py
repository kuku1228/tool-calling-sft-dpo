"""可复现的哈希派生工具。

为什么不能用内建 ``hash()``
---------------------------
Python 3.3 起，``str`` / ``bytes`` / ``tuple`` 等类型的 ``hash()`` 默认
**按进程随机化**（PYTHONHASHSEED 机制，用于防御哈希碰撞攻击）。后果::

    $ python -c "print(hash(('s','x',0)))"
    -2668731556822207002
    $ python -c "print(hash(('s','x',0)))"
    -8349074676084224989      # 同一个表达式，不同的值

所以「用 ``hash(...)`` 当随机种子」会让**同一次实验的两次运行产生不同数据**，
而且不会有任何报错 —— 只是结果悄悄变了。

本项目开发期遇到过：偏好数据的生成用了 ``hash((seed, name, idx))``，
导致每次运行得到的偏好对内容不同（数量一样，内容哈希不同），
mock 流水线的「注入一致性」也在 94%~98% 之间无故抖动。
这会让消融实验失去可比性 —— **你以为在重复同一个实验，其实每次都在换数据**。

正确做法
--------
用 ``hashlib`` 做确定性哈希。``hashlib.md5`` 不受 PYTHONHASHSEED 影响，
跨进程、跨平台、跨 Python 版本都稳定。

（这里不需要密码学安全性，md5 足够且快。若担心碰撞，可用 sha256，
但代价是略慢；本项目输入空间很小，md5 完全够用。）
"""

from __future__ import annotations

import hashlib
from typing import Any

#: 种子空间上限。派生出的种子会落在这个范围内，便于直接喂给 random.Random。
SEED_SPACE = 0xFFFFFFFF


def stable_seed(*parts: Any) -> int:
    """由任意一组值派生一个**确定性**的整数种子。

    与内建 ``hash()`` 的关键差别：结果不受进程、平台、PYTHONHASHSEED 影响。

    >>> stable_seed("a", "b", 0) == stable_seed("a", "b", 0)
    True
    >>> stable_seed(1, 2) != stable_seed(2, 1)
    True
    >>> 0 <= stable_seed("x") <= SEED_SPACE
    True
    >>> # 分隔符防止 ('ab','c') 与 ('a','bc') 撞成同一种子
    >>> stable_seed("ab", "c") != stable_seed("a", "bc")
    True
    """
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return int(digest[:8], 16) % (SEED_SPACE + 1)
