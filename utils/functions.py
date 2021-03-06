# パッケージのimport
import random
import time
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from collections import OrderedDict



# Setup seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

def make_datapath_list():
    """学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。 """

    train_img_list = list()  # 画像ファイルパスを格納

    for img_idx in range(200):
        img_path = "./data/img_78/img_7_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

        img_path = "./data/img_78/img_8_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

    return train_img_list


# ネットワークの初期化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# モデルを学習させる関数を作成
def train_model(G, D, dataloader, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # 最適化手法の設定
    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])

    # 誤差関数を定義
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # パラメータをハードコーディング
    z_dim = 20
    mini_batch_size = 64

    # ネットワークをGPUへ
    if torch.cuda.device_count()>1:
        # 複数枚使える時複数枚使う
        print("Let's use {} GPUs".format(torch.cuda.device_count()))
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)

    G.to(device)
    D.to(device)

    G.train()  # モデルを訓練モードに
    D.train()  # モデルを訓練モードに

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # イテレーションカウンタをセット
    iteration = 1
    logs = []
    with tqdm(range(num_epochs)) as pbar:
        # epochのループ
        for epoch in pbar:

            # 開始時刻を保存
            t_epoch_start = time.time()
            epoch_g_loss = 0.0  # epochの損失和
            epoch_d_loss = 0.0  # epochの損失和

            pbar.set_description(f"[Epoch {epoch + 1} /{num_epochs}]")

            # データローダーからminibatchずつ取り出すループ
            for imges in dataloader:

                # --------------------
                # 1. Discriminatorの学習
                # --------------------
                # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
                if imges.size()[0] == 1:
                    continue

                # GPUが使えるならGPUにデータを送る
                imges = imges.to(device)

                # 正解ラベルと偽ラベルを作成
                # epochの最後のイテレーションはミニバッチの数が少なくなる
                mini_batch_size = imges.size()[0]
                label_real = torch.full((mini_batch_size,), 1).to(device)
                label_fake = torch.full((mini_batch_size,), 0).to(device)

                # 真の画像を判定
                d_out_real, _ = D(imges)

                # 偽の画像を生成して判定
                input_z = torch.randn(mini_batch_size, z_dim).to(device)
                input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
                fake_images = G(input_z)
                d_out_fake, _ = D(fake_images)

                # 誤差を計算
                label_real = label_real.type_as(d_out_real.view(-1))
                label_fake = label_fake.type_as(d_out_fake.view(-1))
                d_loss_real = criterion(d_out_real.view(-1), label_real)
                d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
                d_loss = d_loss_real + d_loss_fake

                # バックプロパゲーション
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()

                d_loss.backward()
                d_optimizer.step()

                # --------------------
                # 2. Generatorの学習
                # --------------------
                # 偽の画像を生成して判定
                input_z = torch.randn(mini_batch_size, z_dim).to(device)
                input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
                fake_images = G(input_z)
                d_out_fake, _ = D(fake_images)

                # 誤差を計算
                g_loss = criterion(d_out_fake.view(-1), label_real)

                # バックプロパゲーション
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                # --------------------
                # 3. 記録
                # --------------------
                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                iteration += 1

            # epochのphaseごとのlossと正解率
            t_epoch_finish = time.time()

            pbar.set_postfix(OrderedDict(timer=f"{t_epoch_finish - t_epoch_start:.4f}",
                                            Epoch_D_Loss=f"{epoch_d_loss/batch_size:.4f}",
                                            Epoch_G_Loss=f"{epoch_g_loss/batch_size:.4f}"),)
            t_epoch_start = time.time()


        print("総イテレーション回数:", iteration)

        return G, D


def efficient_make_datapath_list():
    """学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。 """

    train_img_list = list()  # 画像ファイルパスを格納

    for img_idx in range(200):
        img_path = "./data/img_78_28size/img_7_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

        img_path = "./data/img_78_28size/img_8_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

    return train_img_list


def efficient_train_model(G, D, E, dataloader, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # 最適化手法の設定
    lr_ge = 0.0001
    lr_d = 0.0001/4
    beta1, beta2 = 0.5, 0.999
    g_optimizer = torch.optim.Adam(G.parameters(), lr_ge, [beta1, beta2])
    e_optimizer = torch.optim.Adam(E.parameters(), lr_ge, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), lr_d, [beta1, beta2])

    # 誤差関数を定義
    # BCEWithLogitsLossは入力にシグモイド（logit）をかけてから、
    # バイナリークロスエントロピーを計算
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # パラメータをハードコーディング
    z_dim = 20
    mini_batch_size = 64

    # ネットワークをGPUへ
    if torch.cuda.device_count()>1:
      # 複数枚使える時複数枚使う
        print("Let's use {} GPUs".format(torch.cuda.device_count()))
        G = nn.DataParallel(G)
        E = nn.DataParallel(E)
        D = nn.DataParallel(D)

    G.to(device)
    E.to(device)
    D.to(device)

    G.train()  # モデルを訓練モードに
    E.train()  # モデルを訓練モードに
    D.train()  # モデルを訓練モードに

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # イテレーションカウンタをセット
    iteration = 1
    logs = []

    with tqdm(range(num_epochs)) as pbar:
    # epochのループ
        for epoch in pbar:

            # 開始時刻を保存
            t_epoch_start = time.time()
            epoch_g_loss = 0.0  # epochの損失和
            epoch_e_loss = 0.0  # epochの損失和
            epoch_d_loss = 0.0  # epochの損失和

            pbar.set_description(f"[Epoch {epoch + 1} /{num_epochs}]")

            # データローダーからminibatchずつ取り出すループ
            for imges in dataloader:

                # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
                if imges.size()[0] == 1:
                    continue

                # ミニバッチサイズの1もしくは0のラベル役のテンソルを作成
                # 正解ラベルと偽ラベルを作成
                # epochの最後のイテレーションはミニバッチの数が少なくなる
                mini_batch_size = imges.size()[0]
                label_real = torch.full((mini_batch_size,), 1).to(device)
                label_fake = torch.full((mini_batch_size,), 0).to(device)

                # GPUが使えるならGPUにデータを送る
                imges = imges.to(device)

                # --------------------
                # 1. Discriminatorの学習
                # --------------------
                # 真の画像を判定　
                z_out_real = E(imges)
                d_out_real, _ = D(imges, z_out_real)

                # 偽の画像を生成して判定
                input_z = torch.randn(mini_batch_size, z_dim).to(device)
                fake_images = G(input_z)
                d_out_fake, _ = D(fake_images, input_z)

                # 誤差を計算
                label_real = label_real.type_as(d_out_real.view(-1))
                label_fake = label_fake.type_as(d_out_fake.view(-1))
                d_loss_real = criterion(d_out_real.view(-1), label_real)
                d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
                d_loss = d_loss_real + d_loss_fake

                # バックプロパゲーション
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                # --------------------
                # 2. Generatorの学習
                # --------------------
                # 偽の画像を生成して判定
                input_z = torch.randn(mini_batch_size, z_dim).to(device)
                fake_images = G(input_z)
                d_out_fake, _ = D(fake_images, input_z)

                # 誤差を計算
                g_loss = criterion(d_out_fake.view(-1), label_real)

                # バックプロパゲーション
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                # --------------------
                # 3. Encoderの学習
                # --------------------
                # 真の画像のzを推定
                z_out_real = E(imges)
                d_out_real, _ = D(imges, z_out_real)

                # 誤差を計算
                e_loss = criterion(d_out_real.view(-1), label_fake)

                # バックプロパゲーション
                e_optimizer.zero_grad()
                e_loss.backward()
                e_optimizer.step()

                # --------------------
                # 4. 記録
                # --------------------
                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                epoch_e_loss += e_loss.item()
                iteration += 1

            # epochのphaseごとのlossと正解率
            t_epoch_finish = time.time()

            pbar.set_postfix(OrderedDict(timer=f"{t_epoch_finish - t_epoch_start:.4f}",
                                         Epoch_D_Loss=f"{epoch_d_loss/batch_size:.4f}",
                                         Epoch_G_Loss=f"{epoch_g_loss/batch_size:.4f}"),
                                         Epoch_E_Loss=f"{epoch_e_loss/batch_size:.4f}",)
            t_epoch_start = time.time()

        print("総イテレーション回数:", iteration)

        return G, D, E


# ネットワークの初期化
def efficient_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        # 全結合層Linearの初期化
        m.bias.data.fill_(0)