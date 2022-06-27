# パッケージのimport
import random
import numpy as np

import torch

import utils.models as models
import utils.functions as functions
import utils.preprocessing as preprocessing


# Setup seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

G = models.Generator(z_dim=20, image_size=64)
# 動作確認
D = models.Discriminator(z_dim=20, image_size=64)

# DataLoaderの作成と動作確認
# ファイルリストを作成
train_img_list=functions.make_datapath_list()

# Datasetを作成
mean = (0.5,)
std = (0.5,)
train_dataset = preprocessing.GAN_Img_Dataset(file_list=train_img_list, transform=preprocessing.ImageTransform(mean, std))

# DataLoaderを作成
batch_size = 64

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 動作の確認
batch_iterator = iter(train_dataloader)  # イテレータに変換
imges = next(batch_iterator)  # 1番目の要素を取り出す

# 初期化の実施
G.apply(functions.weights_init)
D.apply(functions.weights_init)

print("ネットワークの初期化完了")

# 学習・検証を実行する
# 8分ほどかかる
num_epochs = 300
G_update, D_update = functions.train_model(G, D, dataloader=train_dataloader, num_epochs=num_epochs)
