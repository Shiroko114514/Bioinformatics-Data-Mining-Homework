使用方法
1.训练模型：运行 python wrapper.py，它会加载训练/测试数据，训练模型并保存。
2.预测序列：运行 python predict.py "<DNA序列>" [模型路径] [阈值]，例如：
    python predict.py "ATCGATCGATCGAAGGTAAGTATCGGCATCGATCGATCG"
3.输出预测的剪接位点位置、分数和上下文。