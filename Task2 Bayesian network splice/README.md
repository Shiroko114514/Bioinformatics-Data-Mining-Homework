## 运行说明

### 1. 基本依赖

- Python 3.8+
- 无外部依赖，仅标准库

### 2. 命令行Demo（默认使用真实数据）

```bash
cd "Task2 Bayesian network splice"
python3 splice_main.py --demo
```

### 3. 输入一段序列预测 

```bashm
python3 splice_main.py --predict "ACGTTGGTAGT..." --site donor --threshold 0.0
```

可选参数：
- `--site`：`donor` 或 `acceptor`（默认为 `donor`）
- `--threshold`：模型分数阈值（默认为 `0.0`）
- `--window`：`9` 或 `23`，若未设置则根据 `site` 自动选择
- `--no-real-data`：使用合成数据而非真实集

## 数据集路径

- `Training and testing datasets/Training Set/`：训练（GenBank `.TXT`）
- `Training and testing datasets/Testing Set/`：测试（FASTA-style `.TXT` / `.txt`）

## 代码验证

- 语法检查：`python3 -m py_compile *.py`
- 演示预测：
  `python3 splice_main.py --predict GTGAGTGGTA --site donor --threshold -5.0`

输出示例：
```
pos\tscore\twindow
4\t-1.0581\tTGAGTGGTA
```

## 说明

- `donor` 模型窗口 `9`：GT 在窗口位置 `3-4`
- `acceptor` 模型窗口 `23`：AG 在窗口位置 `20-21`
- `generate_negative_samples` 保持正负平衡并避免已知正例


欢迎依据模型需求继续优化 `splice_model.py` 中 `greedy-bic` 结构搜索、权重参数和性能。
