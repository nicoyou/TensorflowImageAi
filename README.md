# Tensorflow Image
### 基本的な画像関連AIの学習と推論が簡単にできるライブラリ。
このライブラリでは画像の分類、回帰、マルチラベル分類、pix2pixのディープラーニングを 2, 3 行のコードで容易に実行することができます。
学習に使用した画像数や学習履歴などの情報も保存されるため、後から確認したりグラフに表示することができます。また、続きから学習を再開することも可能です。
そのため、学習の中止と再開を何度も繰り返しても一つの連続した学習履歴として表示できます。

テストデータに対する推論結果を、正解のパラメータと比較できる図として表示することができます。

ディープラーニングで設定することのできる各パラメータは汎用的な値で最適化されています。

[![PyPI](https://img.shields.io/pypi/v/tensorflow-image)](https://pypi.org/project/tensorflow-image/)
![Python versions](https://img.shields.io/pypi/pyversions/tensorflow-image)

---

## インストール方法
```bash
pip install tensorflow-image
```


## 使用方法
"./dataset/" ディレクトリに教師データとなる画像を格納します。

分類問題ではクラスごとにディレクトリを分けて画像を格納することで、ディレクトリ名を分類の名前として認識します。


回帰問題と多ラベル分類では、各画像の情報を格納した csvファイル のパスを指定する必要があります。

それぞれの csvファイル 内構成は以下のようになります。
filename は この csvファイル からの相対パスになります。


### 回帰問題
|  filename       |  class  |
| --------------- | ------- |
|  ./image/a.png  |  0.3    |
|  ./image/b.png  |  0.6    |

### 多ラベル分類
|  filename       |  labels                   |
| --------------- | ------------------------- |
|  ./image/a.png  |  label_a,label_c,label_d  |
|  ./image/b.png  |  label_b                  |


AIをトレーニングする。
```python
import tensorflow_image as tfimg

# 画像分類
icai = tfimg.ImageClassificationAi("model_name_classification")
icai.train_model("./dataset/", epochs=6, model_type=tfimg.ModelType.efficient_net_v2_b0, trainable=True)

# 画像の回帰問題
rai = tfimg.ImageRegressionAi("model_name_regression")
rai.train_model("./dataset/data.csv", epochs=6, model_type=tfimg.ModelType.efficient_net_v2_b0, trainable=True)

# 多ラベル分類
mlai = tfimg.ImageMultiLabelAi("model_name_multilabel")
mlai.train_model("./dataset/data.csv", epochs=6, model_type=tfimg.ModelType.efficient_net_v2_b0, trainable=True)
```

学習済みモデルを読み込んで推論する。
```python
import tensorflow_image as tfimg

# 画像分類
icai = tfimg.ImageClassificationAi("model_name_classification")
icai.load_model()
result = icai.predict("./dataset/sample.png")   # 推論する
print(icai.result_to_classname(result))

# 画像の回帰問題
rai = tfimg.ImageRegressionAi("model_name_regression")
rai.load_model()
result = rai.predict("./dataset/sample.png")    # 推論する

# 多ラベル分類
mlai = tfimg.ImageMultiLabelAi("model_name_multilabel")
mlai.load_model()
result = mlai.predict("./dataset/sample.png")   # 推論する
print(mlai.result_to_label_dict(result))
print(mlai.result_to_labelname(result, 0.5))
```

---

## 対応モデル
- MobileNetV2
- VGG16
- EfficientNetV2
- ResNet-RS


ResNet-RS を使用するには以下のリポジトリを追加で pip install する必要があります
> git+https://github.com/sebastian-sz/resnet-rs-keras@main

---

## 制作者
Nicoyou

[@NicoyouSoft](https://twitter.com/NicoyouSoft)
