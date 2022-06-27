# パッケージのimport
import random
import numpy as np

import torch

import utils.models as models
import utils.functions as functions
import utils.preprocessing as preprocessing



# Setup seeds
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# 動作確認
# DataLoaderを作成
batch_size = 64
num_epochs = 1500

G = models.EfficientGenerator(z_dim=20)
G.train()

# 動作確認
D = models.EfficientDiscriminator(z_dim=20)

# 動作確認
E = models.EfficientEncoder(z_dim=20)

# ファイルリストを作成
train_img_list=functions.efficient_make_datapath_list()

# Datasetを作成
mean = (0.5,)
std = (0.5,)
train_dataset = preprocessing.Efficient_GAN_Img_Dataset(
    file_list=train_img_list, transform=preprocessing.EfficientImageTransform(mean, std))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# モデルを学習させる関数を作成
# 初期化の実施
G.apply(functions.efficient_weights_init)
E.apply(functions.efficient_weights_init)
D.apply(functions.efficient_weights_init)

print("ネットワークの初期化完了")

# 学習・検証を実行する
# 15分ほどかかる
G_update, D_update, E_update = functions.efficient_train_model(G, D, E, dataloader=train_dataloader, num_epochs=num_epochs)