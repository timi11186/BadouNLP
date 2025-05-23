# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
zhou wen jian


与极度欠拟合相比，这里使用的是三层神经网络，而不是一层神经网络。


"""

# 1. 建立模型
class TorchModel(nn.Module):
    # 1.1 接收两个参数 本身以及个数
    def __init__(self, input_size):
        super(TorchModel,self).__init__()
        self.linear1 = nn.Linear(input_size, 32)  # 第一层：输入层到隐藏层
        self.linear2 = nn.Linear(32, 16)          # 第二层：隐藏层到隐藏层
        self.linear3 = nn.Linear(16, 5)           # 第三层：隐藏层到输出层
        self.relu = nn.ReLU()  # 使用ReLU作为中间层激活函数
        self.softmax = nn.Softmax()  # 使用Softmax作为输出层激活函数
        self.loss = nn.functional.cross_entropy
    
    # 1.2 当输入真实标签，返回 loss 值：无真实标签，返回预测值
    def forward(self,x,y=None):
        x = self.relu(self.linear1(x))  # 第一层 + ReLU  这里使用ReLU作为中间层激活函数
        x = self.relu(self.linear2(x))  # 第二层 + ReLU
        x = self.linear3(x)             # 第三层（输出层不需要激活函数）
        if y is not None:
            return self.loss(x, y)  # cross_entropy会自动应用softmax ，判断原始值与预测值的差异
        else:
            return self.softmax(x)  # 预测时手动应用softmax，

# 2. 建立样本进行模型测试
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x) # 最大值的索引
    return x,y

# 3. 建立数据集
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x,y = build_sample() # 生成随机样本
        X.append(x) # 加入到 X 输入样本中
        Y.append(y) # 加入到 Y 输出样本中，输出是标量，
        #使用[y]的形式转化为二维张量，为了统一处理 ，
        # 使用 【y】 的形式转化为二维张量在该训练中会报错
    # 将 python 的数组类型转化为，pyThorch 的张量类型
    # 张量是 pyTorch 中处理数据的基本格式
    return torch.FloatTensor(X),torch.LongTensor(Y)

# 3. 测试代码
# 用于测试每轮模型的准确率
def evaluate(model):
    model.eval()  # 设置为评估模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次测试预测集中共有%d个样本，其中" % len(x))
    
    # 统计每个类别的样本数
    for i in range(5):
        print(f"- 第{i+1}类样本 {(y == i).sum().item()}个")
    
    correct, wrong = 0, 0
    with torch.no_grad():  # 禁用梯度计算
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):
            # 获取预测的类别（概率最大的那个）
            pred_class = torch.argmax(y_p).item()
            # 获取真实类别
            true_class = y_t.item()
            
            # 比较预测类别和真实类别
            if pred_class == true_class:
                correct += 1
            else:
                wrong += 1
    
    # 打印预测结果
    print(f"正确预测：{correct}个")
    print(f"错误预测：{wrong}个")
    print(f"准确率：{correct/(correct+wrong):.2%}")
    
    return correct / (correct + wrong)  # 返回准确率

# 4. 正式使用
def main():
    # 配置参数
    epoch_num = 100        # 增加训练轮数
    batch_size = 32        # 调整batch size为更常用的值
    train_sample = 10000   # 增加训练样本数量
    learning_rate = 0.0005 # 降低学习率，使训练更稳定
    input_size = 5  # 输入向量维度，因为是要用五维作为输入 x
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 开始训练
    for epoch in range(epoch_num):
        model.train() # 开始梯度学习
        watch_loss = [] 
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size] # 切片获取对应的x
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size] # 切片获取对应的y
            loss = model(x,y) # 使用模型，获取 loss
            loss.backward() # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item()) # 记录一下 loss
        
        avg_loss = np.mean(watch_loss) # 计算平均loss
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append((acc, avg_loss))

    #保存模型
    torch.save(model.state_dict(), "model.bin")

    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, vec_list):
    # 加载模型
    model = TorchModel(input_size=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    
    # 转换为张量
    vec_tensor = torch.tensor(vec_list, dtype=torch.float32)
    
    # 获取预测结果
    with torch.no_grad():
        outputs = model(vec_tensor)
        probabilities = torch.softmax(outputs, dim=1)  # 应用softmax获取概率
        predicted_classes = torch.argmax(probabilities, dim=1)  # 获取最大概率的类别
        max_probabilities = torch.max(probabilities, dim=1)[0]  # 获取最大概率值
    
    # 打印每个样本的结果
    for i, (vec, pred_class, prob) in enumerate(zip(vec_list, predicted_classes, max_probabilities)):
        print(f"样本 {i+1}:")
        print(f"输入：{vec}")
        print(f"预测类别：{pred_class.item() + 1}")
        print(f"概率值：{prob.item():.4f}")
        print("-" * 30)


if __name__ == "__main__":
    # main()
        # (AI改：原来的简单测试集 -- 现在的更全面的测试集)
    test_vec = [
        # 1. 明显最大值在最后一维
        [0.1, 0.2, 0.3, 0.4, 0.9],  # 第5类
        # 2. 明显最大值在第一维
        [0.95, 0.1, 0.2, 0.3, 0.4],  # 第1类
        # 3. 最大值在中间维
        [0.2, 0.3, 0.95, 0.1, 0.4],  # 第3类
        # 4. 数值接近的情况
        [0.45, 0.44, 0.43, 0.42, 0.46],  # 第5类
        # 5. 数值差异很小的情况
        [0.501, 0.499, 0.498, 0.497, 0.496],  # 第1类
        # 6. 包含0和1的边界情况
        [0.0, 0.1, 0.2, 0.3, 1.0],  # 第5类
        # 7. 所有值都很小
        [0.01, 0.02, 0.03, 0.04, 0.05],  # 第5类
        # 8. 所有值都很大
        [0.95, 0.96, 0.97, 0.98, 0.99],  # 第5类
        # 9. 随机分布
        [0.123, 0.456, 0.789, 0.321, 0.654],  # 第3类
        # 10. 最大值在第二维
        [0.3, 0.9, 0.4, 0.5, 0.6]  # 第2类
        ]
    predict("model.bin", test_vec)
