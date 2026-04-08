# SVM Splice Site Predictor

# SVM Splice Site Predictor

## 使用方法

### 1. 训练模型

运行 `python wrapper.py`，它会加载训练/测试数据，训练模型并保存到当前目录下的 `svm_splice_site.pkl`。

```bash
cd "Task3 Svm splice site"
python3 wrapper.py
```

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

### 3. 输出内容

程序会输出预测的剪接位点位置、分数和上下文。

## 目录说明

- `wrapper.py`：训练模型并保存
- `predict.py`：加载模型并对输入 DNA 序列做扫描预测
- `splice_data.py`：数据读取、正负样本构造
- `splice_features.py`：特征提取
- `splice_model.py`：SVM 模型训练、预测、扫描
- `splice_eval.py`：消融实验和模型对比
- `splice_main.py`：主入口 demo
- `svm_splice_site.py`：兼容入口，保留旧导入方式
- `splice_utils.py`：公共常量和工具函数

## 说明

- 默认 donor 窗口长度为 `9`，其中 `GT` 位于窗口位置 `3-4`
- 默认 acceptor 窗口长度为 `23`，其中 `AG` 位于窗口位置 `20-21`
- 代码依赖 `numpy` 和 `scikit-learn`

## 额外入口

如果你想直接看完整演示，也可以运行：

```bash
python3 splice_main.py
```
