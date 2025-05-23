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

"""

# 1. 建立模型
class TorchModel(nn.Module):
    # 1.1 接收两个参数 本身以及个数
    def __init__(self, input_size):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size,5) # 线性层,输出5个数,因为分类有五个
        # 激活函数 activation 为了让线性层有非线性分布
        self.activation = torch.sigmoid ## sigmoid归一化函数
        self.loss = nn.functional.cross_entropy # 交叉熵损失函数
    
    # 1.2 当输入真实标签，返回 loss 值：无真实标签，返回预测值
    def forward(self,x,y=None):
        x = self.linear(x) # 将x 矩阵 或者 张量传入线性层进行输出对应的预测值
        y_pred = self.activation(x) # 使用激活函数 使得线性层的预测值有更加相关预测
        if y is not None:
            return self.loss(y_pred,y) # 返回预测值与真实值的损失 loss
        else:
            return y_pred  # 否则输出预测结果

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
    model.eval() # 不需要梯度
    test_sample_num = 100 # 测试集长度，个数。
    x,y = build_dataset(test_sample_num)
    print("本次测试预测集中共有%d个样本，其中" % len(x))
    print("- 第一类样本 %d个" % (y == 0).sum())
    print("- 第二类样本 %d个" % (y == 1).sum())
    print("- 第三类样本 %d个" % (y == 2).sum())
    print("- 第四类样本 %d个" % (y == 3).sum())
    print("- 第五类样本 %d个" % (y == 4).sum())

    class_correct = [0] * 5 # 用于记录每个类别的正确次数
    class_wrong = [0] * 5 # 用于记录每个类别的错误次数
    class_total = [0] * 5 # 用于记录每个类别的总次数
    correct, wrong = 0, 0

    # 禁用梯度计算
    with torch.no_grad():
        y_pred = model(x) # 使用模型，获取预测值，输出 1-5 五个类别
        # y_p 是预测值，y_t 是真实值
        for y_p,y_t in zip(y_pred,y): # 与真实值进行比较
            if y_p == y_t:
                class_correct[y_t] += 1 # 正确次数加1
                correct += 1 # 整体正确次数加一
            else:
                class_wrong[y_t] += 1 # 错误次数加 1
                wrong += 1 # 整体错误次数加一
            class_total[y_t] += 1 # 总次数加1
    print("正确次数 %d 。正确率 %f " % (correct, correct / (correct + wrong)))
    # print("- 第一类样本预测正确 %d 个，一类预测正确率 %f" %  (class_correct[1],class_correct[1] / class_wrong[1])
    # print("- 第二类样本预测正确 %d 个，一类预测正确率 %f" %  (class_correct[2],class_correct[2] / class_wrong[2]))
    # print("- 第三类样本预测正确 %d 个，一类预测正确率 %f" %  (class_correct[3],class_correct[3] / class_wrong[3]))
    # print("- 第四类样本预测正确 %d 个，一类预测正确率 %f" %  (class_correct[4],class_correct[4] / class_wrong[4]))
    # print("- 第五类样本预测正确 %d 个，一类预测正确率 %f" %  (class_correct[5],class_correct[5] / class_wrong[5]))
    return correct / (correct + wrong)

# 4. 正式使用
def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度，因为是要用五维作为输入 x
    learning_rate = 0.001  # 学习率
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
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


# if __name__ == "__main__":
#     # main()
#     test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
#                 [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
#                 [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
#                 [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
#     predict("model.bin", test_vec)

main()