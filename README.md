## Usage
　リポジトリの使い方

### 環境構築
以下のどれかを行う

**conda 環境の準備**

```bash
conda env create -f ./env_config/conda_env.yaml -n env_name
```

**pip 環境**

```bash
pip install -r ./env_config/requirements.txt
```

### Datasetの作成

```
make_dataset.py
```

### AnoGANの動かし方

```
python anoGan.py
```

### EfficientGANの動かし方

```
python EfficientGan.py
```



## 出典

**つくりながら学ぶ! PyTorchによる発展ディープラーニング**

[書籍「つくりながら学ぶ! PyTorchによる発展ディープラーニング」（小川雄太郎、マイナビ出版 、19/07/29) ](https://www.amazon.co.jp/dp/4839970254/)

を卒業研究でGANを使った異常検知を行う人むけに改変したリポジトリです。

特に
- 第5章 GANによる画像生成（DCGAN、Self-Attention GAN）
- 第6章 GANによる異常検知（AnoGAN、Efficient GAN)
を中心に改変し、整理したリポジトリです。

そのほかの章については詳しく以下にまとめられています。
[「Qiita記事：PyTorchによる発展ディープラーニング、各章の紹介」](https://qiita.com/sugulu/items/07253d12b1fc72e16aba)