# SVM Splice Site Predictor

## 使用方法

### 0. 环境依赖

- Python 3.13+
- `numpy`
- `scikit-learn`

### 1. 训练模型

运行 `python wrapper.py`，它会加载训练/测试数据，训练模型并保存到当前目录下的 `svm_splice_site.pkl`。

```bash
cd "Task3 Svm splice site"
python3 wrapper.py
```

训练过程使用真实数据集中的 donor 位点与对应负样本，并在生成负样本时保持数量平衡；如果数据不足，程序会直接报错而不是静默返回不完整结果。

### 2. 预测序列

运行 `python predict.py "<DNA序列>" [模型路径] [阈值]`。

例如：

```bash
python3 predict.py "ATCGATCGATCGAAGGTAAGTATCGGCATCGATCGATCG"
```

也可以显式指定模型路径和阈值：

```bash
python3 predict.py "ATCGATCGATCGAAGGTAAGTATCGGCATCGATCGATCG" svm_splice_site.pkl 0.0
```

如果模型文件不存在，程序会提示先运行 `wrapper.py`。

### 3. 输出内容

程序会输出预测的剪接位点位置、分数和上下文。

### 4. 完整 demo 与 ROC 图

运行 `python3 splice_main.py` 会执行完整 demo：

- 特征消融实验
- 核函数比较
- WMM/WAM/BN/SVM 四方比较
- 线性 SVM 重要特征
- 5 折交叉验证
- genome scan

同时会在当前目录生成 ROC 图文件 `roc_curves.svg`。

ROC 画图代码被单独拆分到了 `roc_plot.py`，`splice_eval.py` 只负责收集 ROC 数据并调用该模块。

## 目录说明

- `wrapper.py`：训练模型并保存
- `predict.py`：加载模型并对输入 DNA 序列做扫描预测
- `splice_data.py`：数据读取、正负样本构造
- `splice_features.py`：特征提取
- `splice_model.py`：SVM 模型训练、预测、扫描
- `splice_eval.py`：消融实验和模型对比
- `roc_plot.py`：独立 ROC 绘图模块，输出 SVG 图
- `splice_main.py`：主入口 demo
- `splice_utils.py`：公共常量和工具函数

## 说明

- 默认 donor 窗口长度为 `9`，其中 `GT` 位于窗口位置 `3-4`
- 默认 acceptor 窗口长度为 `23`，其中 `AG` 位于窗口位置 `20-21`
- 代码依赖 `numpy` 和 `scikit-learn`
- `cross_validate()` 会在每一折内重新拟合特征提取器和分类器，避免 PWM 特征泄漏到验证折中
- `make_donor_negative()` 仅用于合成负样本，不会生成包含 donor `GT` 模式的窗口

## 额外入口

如果你想直接看完整演示，也可以运行：

```bash
python3 splice_main.py
```

如果只想查看 ROC 图，可直接运行 demo 后打开 `roc_curves.svg`。
