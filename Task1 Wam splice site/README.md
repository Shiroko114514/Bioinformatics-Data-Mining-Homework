使用方法
1.训练模型：运行 python wrapper.py，它会加载训练/测试数据，训练模型并保存。
2.预测序列：运行 python predict.py "<DNA序列>" [模型路径] [阈值]，例如：
    python predict.py "ATCGATCGATCGAAGGTAAGTATCGGCATCGATCGATCG"
3.输出预测的剪接位点位置、分数和上下文。

## 更新说明

- `main.py` 现在会比较 `WMM`、`WAM` 和一个依赖图增强的 `Dependency-WAM`。
- 支持 `--site donor` 和 `--site acceptor`，默认是 `donor`。
- 训练和测试数据路径改为基于仓库位置自动解析，不再依赖硬编码绝对路径。
- 当真实数据不可用时，会自动生成 donor 或 acceptor 的合成数据作为回退。
- `main.py` 的 demo 会自动生成 `roc_curves.svg`。
- ROC 绘图逻辑被单独拆到了 `roc_plot.py`。

## ROC 图

运行 `python main.py --demo` 后，会在当前目录生成 `roc_curves.svg`，其中包含 `WMM`、`WAM` 和 `Dependency-WAM` 的 ROC 曲线。

如果只想复用绘图逻辑，可以直接调用 `roc_plot.py` 中的 `plot_roc_curves(...)`。