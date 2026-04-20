## 运行说明

### 1. 基本依赖

- Python 3.8+
- 无外部依赖，仅标准库

### 2. 代码结构

- `roc_plot.py`：独立的 ROC 画图模块，输出 SVG 文件
- `splice_eval.py`：模型对比、ROC 数据收集与调用画图模块
- `splice_main.py`：命令行入口，demo 运行时会自动生成 ROC 图

### 3. 命令行Demo（默认使用真实数据）

```bash
cd "Task2 Bayesian network splice"
python3 splice_main.py --demo
```

运行 demo 后，会在当前目录生成 `roc_curves.svg`，其中包含 WMM、WAM、BN Chow-Liu 和 BN EBN(p=2) 的 ROC 曲线。

### 4. 输入一段序列预测 

```bash
python3 splice_main.py --predict "ACGTTGGTAGT..." --site donor --threshold 0.0
```

可选参数：
- `--site`：`donor` 或 `acceptor`（默认为 `donor`）
- `--threshold`：模型分数阈值（默认为 `0.0`）
- `--window`：`9` 或 `23`，若未设置则根据 `site` 自动选择
- `--structure`：`chain`、`chow-liu` 或 `ebn`，默认 `chow-liu`
- `--max-parents`：EBN 中每个节点最多父节点数，默认 `2`
- `--chi2-threshold`：EBN 依赖边的 χ² 阈值，默认 `6.0`
- `--no-real-data`：使用合成数据而非真实集

## 数据集路径

- `Training and testing datasets/Training Set/`：训练（GenBank `.TXT`）
- `Training and testing datasets/Testing Set/`：测试（FASTA-style `.TXT` / `.txt`）

## 代码验证

- 语法检查：`python3 -m py_compile *.py`
- 演示预测：
  `python3 splice_main.py --predict GTGAGTGGTA --site donor --threshold -5.0`
- 生成 ROC 图：
  `python3 splice_main.py --demo --site donor`

输出示例：
```
pos\tscore\twindow
4\t-1.0581\tTGAGTGGTA
```

## 说明

- `donor` 模型窗口 `9`：GT 在窗口位置 `3-4`
- `acceptor` 模型窗口 `23`：AG 在窗口位置 `20-21`
- `generate_negative_samples` 保持正负平衡并避免已知正例
- `structure='ebn'` 使用 χ² 依赖图构造多父节点 Bayesian Network，并以 `log P(false) - log P(true)` 进行打分

欢迎继续依据模型需求调整 `dependency_threshold`、`max_parents` 和 `threshold` 参数。
