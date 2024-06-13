# deep-learning
深度学习实验这是一个基于传统统计学方法（SARIMAX）和深度学习方法(基于注意力机制的CNN-LSTM模型等)的对比

数据来源：https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data
构建模型：![image](https://github.com/Yanxians/deep-learning/assets/97447243/89ade6ff-45f0-48a7-91e0-048545c0da8f)
模型详细代码在DLmodels文档中。

实验思路：在transformer横空出世之前，RNN、CNN和反向传播是深度学习的三大基石。其中CNN-LSTM模型在时空序列预测中非常火爆。但是现有的模型都有不足。LSTM模型能够记忆较长距离的信息，具有一定的长距离信息挖掘能力。但是在训练过长序列LSTM模型时，有概率出现梯度消失和梯度爆炸的情况，因此也具有局限性。一个想法是结合CNN和LSTM，用CNN提取短序列中的特征信息，然后把信息交给LSTM处理，能够有效降低冗余信息量。然而，标准CNN在特征提取过程中，容易忽略掉一些较少出现的重要特征。而对时序数据来说，这些异常值中的特征也很重要。因此针对CNN-LSTM模型忽略短期特征重要度而导致的重要特征丢失、长期时序规律挖掘有待优化等问题，引入了注意力机制，构成了CNN-LSTM-ATTENTION模型。
大名鼎鼎的transformer就重用了注意力机制。注意力机制是科学家从人的视觉研究中提出的。注意力机制能够提取时间序列的细粒度特征，从而弥补CNN-LSTM模型出现的一些问题。本实验将把注意力机制引入CNN-LSTM中，并评估其效果。

训练结果：
SARIMAX模型训练成果：
![image](https://github.com/Yanxians/deep-learning/assets/97447243/272e6a76-b81f-466b-8f95-d1e9e56f2170)

LSTM模型：
![image](https://github.com/Yanxians/deep-learning/assets/97447243/4b6da3de-3182-4cad-ade3-118993c82ec3)

Seq2seq模型：
![image](https://github.com/Yanxians/deep-learning/assets/97447243/7d55a5c5-0bcd-40ba-bcac-6a430fc080e0)

CNN-LSTM-ATTENTION模型：
![image](https://github.com/Yanxians/deep-learning/assets/97447243/9867eb75-8ac7-43e5-b91a-16fc44276094)

结论：
从这些拟合折线图中我们可以看出，上面四个模型的MSE值分别为42.81、8.70、5.05、6.58。故模型拟合优度：Seq2seq > CNN-LSTM-ATTENTION > LSTM > SARIMAX。不仅图中采用的模型拟合优度排列如此，在多次实验中，95%情况下的模型的拟合优度都是这个顺序，即Seq2seq > CNN-LSTM-ATTENTION > LSTM > SARIMAX。故在时间序列预测中，添加了注意力机制的CNN-LSTM模型的表现的还算优异，不仅表现好于统计学模型，而且好于LSTM模型；与Seq2seq模型相比，CNN-LSTM-ATTENTION还是略逊一筹，但是没有很大的差距。
这次实验中，出乎意料的是Seq2seq模型取得了最好的结果，而且在多次实验中也相对稳定。CNN-LSTM-ATTENTION模型还有一定的进步空间。一方面，本人为深度学习初学者，仅仅是自学，未受到系统学习训练。故在模型搭建和参数调整上可能没有发挥模型的全部潜力，对MSE结果可能有影响。另一方面，前文我们已经提到，CNN-LSTM-ATTENTION模型的强项是他的专注力。因为有CNN和ATTENTION的双重保障，理论上可以让CNN-LSTM-ATTENTION模型在更长时间序列预测中获得相对优异的表现。但是因为数据集大小的限制，我们只记忆了大约1000个数据，并没有发挥出模型的全部实力。
但是作为统计学领域的学生，我觉得这个实验至少给了我们一个思路：当你做预测用传统的统计学模型表现不好时，可以试着使用机器学习、深度学习的方法，说不定会有奇妙的发现。统计学也要多多与时俱进，积极学习深度学习的内容，将深度学习应用到统计学的各个领域。

