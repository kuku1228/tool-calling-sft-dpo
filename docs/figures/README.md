# docs/figures

本目录存放报告引用的图表。**由代码生成，不手工放置。**

## 生成方式

```bash
# 1. 错误分布图 + 按难度/可见性分组图
python scripts/eval/analyze_errors.py --label baseline

# 2. 消融对比图 + pass@k/pass^k 曲线
python scripts/eval/collect_results.py

# 3. 训练曲线（在 mock 流水线里会自动生成）
python scripts/train/mock/run_all_mock.py
```

需要 `matplotlib`：`pip install matplotlib`

> **没有 matplotlib 时不会报错**，只是不出图。这是有意的设计——
> 核心库只依赖标准库 + numpy，让没有深度学习栈的机器也能验证流水线。
> 所有出图函数在 matplotlib 缺失时返回 `None` 而不是抛异常。

## 会生成哪些图

| 文件名 | 内容 | 生成者 |
|---|---|---|
| `error_distribution.png` | 错误类型分布饼图 + 失败子类型 Top-8 柱状图 | `analyze_errors.py` |
| `success_by_difficulty.png` | 按 easy/medium/hard 分组的 accuracy 与 pass^k | `analyze_errors.py` |
| `success_by_visibility.png` | 按 seen/unseen 分组（泛化指标） | `analyze_errors.py` |
| `ablation_comparison.png` | 各实验在 accuracy 与 pass^k 上的分组柱状图 | `collect_results.py` |
| `pass_k_curve.png` | 各实验的 pass@k 与 pass^k 随 k 变化的曲线 | `collect_results.py` |
| `training_curves.png` | 训练 loss（左轴）与验证准确率（右轴）双轴曲线 | `mock_pipeline.py` / 训练脚本 |

## 命名约定

- 基线图有两份：带标签的 `error_distribution_baseline.png` 与不带标签的
  `error_distribution.png`（后者供 README 直接引用）。
- mock 产物带 `_mock_` 中缀，**不要**把它们放进正式报告。

## 中文字体

图表默认按平台依次尝试 `Microsoft YaHei` / `SimHei` / `Noto Sans CJK SC` 等常见中文字体。

**找不到中文字体时会自动退回英文标签**，而不是画出一堆方框。
Linux 服务器上如果发现标签全是方框，装一下字体即可：

```bash
# Debian / Ubuntu
apt-get install -y fonts-noto-cjk
# 然后清掉 matplotlib 字体缓存
rm -rf ~/.cache/matplotlib
```
