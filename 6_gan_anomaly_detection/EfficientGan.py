# パッケージのimport
import random
import math
import time
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

import utils.models as models
import utils.functions as functions



# Setup seeds
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# 動作確認
# import matplotlib.pyplot as plt

G = models.EfficientGenerator(z_dim=20)
G.train()

# 動作確認
D = models.EfficientDiscriminator(z_dim=20)

# 動作確認
E = models.EfficientEncoder(z_dim=20)

# # 入力する乱数
# # バッチノーマライゼーションがあるのでミニバッチ数は2以上
# input_z = torch.randn(2, 20)

# # 偽画像を出力
# fake_images = G(input_z)  # torch.Size([2, 1, 28, 28])
# img_transformed = fake_images[0][0].detach().numpy()
# plt.imshow(img_transformed, 'gray')
# plt.show()

# # 偽画像を生成
# input_z = torch.randn(2, 20)
# fake_images = G(input_z)

# # 偽画像をDに入力
# d_out, _ = D(fake_images, input_z)

# # 出力d_outにSigmoidをかけて0から1に変換
# print(nn.Sigmoid()(d_out))

# # 入力する画像データ
# x = fake_images  # fake_imagesは上のGで作成したもの

# # 画像からzをEncode
# z = E(x)

# print(z.shape)
# print(z)


# DataLoaderの作成と動作確認

# ファイルリストを作成
train_img_list=functions.efficient_make_datapath_list()

# Datasetを作成
mean = (0.5,)
std = (0.5,)
train_dataset = functions.Efficient_GAN_Img_Dataset(
    file_list=train_img_list, transform=functions.EfficientImageTransform(mean, std))

# DataLoaderを作成
batch_size = 64

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 動作の確認
# batch_iterator = iter(train_dataloader)  # イテレータに変換
# imges = next(batch_iterator)  # 1番目の要素を取り出す
# print(imges.size())  # torch.Size([64, 1, 64, 64])


# モデルを学習させる関数を作成

# 初期化の実施
G.apply(functions.efficient_weights_init)
E.apply(functions.efficient_weights_init)
D.apply(functions.efficient_weights_init)

print("ネットワークの初期化完了")

# 学習・検証を実行する
# 15分ほどかかる
num_epochs = 1500
G_update, D_update, E_update = functions.efficient_train_model(G, D, E, dataloader=train_dataloader, num_epochs=num_epochs)