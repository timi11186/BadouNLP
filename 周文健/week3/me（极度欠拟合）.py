import torch
import torch.nn as nn
import numpy as np
import random
import json


"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。

"""
# 1.定义模型
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__() #继承父类
        # 第一个参数表示你需要多少个向量，第二个参数表示每个向量的大小，第三个参数表示指定一个下标作为全 0 向量
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0) #embedding层
        # 使用 RNN 层可以获取到每个字符的信息
        self.rnn = nn.RNN(vector_dim, 100, batch_first=True) #rnn层 
        self.classify = nn.Linear(100, 1) #修改：输出维度改为1，因为我们只需要预测一个值
        self.activation = torch.sigmoid #sigmoid归一化函数
        self.loss = nn.functional.mse_loss #loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x) # 经过embedding层，将字符串转换为向量
        # x = x.transpose(1, 2) # 将向量行列对换，保证信息均匀分布在维度上 ！！！1. 第一次 bug
        # x = self.rnn(x) # 经过rnn层，获取每个字符的注册信息，只使用输出序列，忽略隐藏状态 ！！！2. 第二次 bug
        x, _ = self.rnn(x) # 经过rnn层，获取每个字符的注册信息，只使用输出序列，忽略隐藏状态
        # x = x.squeeze() # 过一次 squeeze 层 ！！！3. 第三次 bug：维度不匹配
        # 修改：只取最后一个时间步的输出，因为我们要预测的是a出现的位置
        x = x[:, -1, :]  # 取最后一个时间步的输出，形状变为 [batch_size, hidden_size]
        x = self.classify(x) # 经过线性层，输出形状为 [batch_size, 1]
        y_pred = self.activation(x) # 经过sigmoid归一化函数，形状保持 [batch_size, 1]
        if y is not None:
            # 确保y的维度与y_pred匹配
            y = y.float()  # 将y转换为float类型
            y = y.view(-1, 1)  # 将y的形状调整为 [batch_size, 1]
            return self.loss(y_pred, y) # 如果输入了真实标签，返回loss值
        else:
            return y_pred # 如果输入了真实标签，返回预测值

# 2. 字符集
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1
    vocab['unk'] = len(vocab)
    return vocab


# 3. 随机生成一个样本
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if set("a") & set(x):
        y = x.index("a")
    else:
        y = -1
    x = [vocab.get(word, vocab['unk']) for word in x] # 将字符串转换为序号，为了做embedding
    return x, y

# 4. 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 5. 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

# 6. 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval() # 设置模型为评估模式
    x, y = build_dataset(200, vocab, sample_length)
    print("本次预测集中共有%d个样本"%len(y))
    correct, wrong = 0, 0
    with torch.no_grad(): # 不计算梯度
        y_pred = model(x)
        y_pred = y_pred.squeeze()  # 将预测结果从 [batch_size, 1] 变为 [batch_size]
        y = y.float()  # 将真实标签转换为float类型
        for y_p, y_t in zip(y_pred, y):
            # 修改：使用张量比较而不是转换为标量
            if (y_p < 0.5 and y_t == 0) or (y_p >= 0.5 and y_t == 1):
                correct += 1
            else:
                wrong += 1
    # 计算准确率
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


# 7. 训练代码
def main():
    #配置参数
    epoch_num = 100        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率

    # 建立字符集
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)

    # 建立优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化训练日志
    log = []  # 用于记录每轮的准确率和loss

    # 训练模型
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
        print("第%d轮训练完成，准确率：%f" % (epoch + 1, acc))

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


#8.使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i])) #打印结果


if __name__ == "__main__":
    main()
    # test_strings = ["fnvfee", "wz你dfg", "rqwdeg", "n我kwww"]
    # predict("model.pth", "vocab.json", test_strings)
