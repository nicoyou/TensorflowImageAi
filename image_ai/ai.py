import abc
import os
from pathlib import Path
from typing import Any, Callable, Final

from . import define

os.environ["PATH"] += ";" + str(define.CURRENT_DIR / "dll")     # 環境変数に一時的に dll のパスを追加する
os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={str(define.CURRENT_DIR / 'CUDA')}"

import matplotlib.pyplot as plt
import nlib3
import tensorflow as tf
from PIL import ImageFile

from . import tf_callback

__version__: Final[str] = "1.2.0"
MODEL_DIR: Final[Path] = define.CURRENT_DIR / "models"
MODEL_FILE: Final[str] = "model"


def model_required(func: Callable) -> Callable:
    """モデルが定義されている場合のみ実行するデコレーター
    """
    def wrapper(self, *args, **kwargs):
        if self.model is None:
            nlib3.print_error_log("モデルが初期化されていません")
            return None
        return func(self, *args, **kwargs)

    return wrapper


class Ai(metaclass=abc.ABCMeta):
    """画像分類系AIの根底クラス
    """
    MODEL_DATA_VERSION: Final[int] = 6

    def __init__(self, ai_type: define.AiType, model_name: str) -> None:
        """管理するAIの初期設定を行う

        Args:
            ai_type: 管理するAIの種類
            model_name: 管理するAIの名前 ( 保存、読み込み用 )
        """
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # 高速にロードできない画像も読み込む
        self.model = None
        self.model_data = None
        self.model_name = model_name
        self.ai_type = ai_type
        self.img_height = 224
        self.img_width = 224
        return

    def preprocess_image(self, img_path: Path | str, normalize: bool = False) -> tf.Tensor:
        """画像を読み込んでテンソルに変換する

        Args:
            img_path: 読み込む画像のファイルパス
            normalize: 0 ～ 1 の範囲に正規化するかどうか

        Returns:
            画像のテンソル
        """
        img_raw = tf.io.read_file(str(img_path))
        image = tf.image.decode_image(img_raw, channels=3)
        image = tf.image.resize(image, (self.img_height, self.img_width))
        if normalize:
            image /= 255.0                  # normalize to [0,1] range
        image = tf.expand_dims(image, 0)    # 次元を一つ増やしてバッチ化する
        return image

    def pillow_image_to_tf_image(self, image: Any, normalize: bool = False) -> tf.Tensor:
        """pillowで読み込んだ画像をテンソルに変換する

        Args:
            image: pillowイメージ
            normalize: 0 ～ 1 の範囲に正規化するかどうか

        Returns:
            画像のテンソル
        """
        tf.keras.preprocessing.image.img_to_array(image)
        image = tf.image.resize(image, (self.img_height, self.img_width))
        if normalize:
            image /= 255.0                  # normalize to [0,1] range
        image = tf.expand_dims(image, 0)    # 次元を一つ増やしてバッチ化する
        return image

    def get_normalize_flag(self, model_type: define.ModelType = define.ModelType.unknown) -> bool:
        """画像の前処理として正規化を実行するかを判断する

        Args:
            model_type:
                正規化が必要かどうかを確認するモデル
                すでにクラスがモデルデータを保持している場合は不要

        Returns:
            正規化が必要かどうか
        """
        normalize = True
        no_normalize_model = [define.ModelType.vgg16_512]
        if (self.model_data is not None and self.model_data[define.AiDataKey.model] in no_normalize_model) or model_type in no_normalize_model:
            normalize = False
        return normalize

    def create_generator(self, normalize: bool) -> tf.keras.preprocessing.image.ImageDataGenerator:
        """データセットの前処理を行うジェネレーターを作成する

        Args:
            normalize: 画像の各ピクセルを0 ～ 1 の間に正規化するかどうか

        Returns:
            ジェネレーター
        """
        params = {
            "horizontal_flip": True,        # 左右を反転する
            "rotation_range": 20,           # 度数法で最大変化時の角度を指定
            "channel_shift_range": 15,      # 各画素値の値を加算・減算する ( 最大値を指定する )
            "height_shift_range": 0.03,     # 中心位置を相対的にずらす ( 元画像の高さ X 値 の範囲内で垂直方向にずらす )
            "width_shift_range": 0.03,
            "validation_split": 0.1,        # 全体に対するテストデータの割合
        }
        if normalize:
            params["rescale"] = 1. / 255
        return tf.keras.preprocessing.image.ImageDataGenerator(**params)

    @abc.abstractmethod
    def create_dataset():
        """訓練用とテスト用のデータセットを作成する"""

    @abc.abstractmethod
    def create_model():
        """AIのモデルを作成する"""

    @abc.abstractmethod
    def count_image_from_dataset():
        """データセットに含まれるクラスごとの画像の数を取得する"""

    def train_model(self, dataset_path: str, epochs: int = 6, batch_size: int = 32, model_type: define.ModelType = define.ModelType.unknown, trainable: bool = False) -> dict:
        """ディープラーニングを実行する

        Args:
            dataset_path: 学習に使用するデータセットのファイルパス
            epochs: 学習を行うエポック数
            batch_size: バッチサイズ
            model_type:
                学習を行うモデルの種類
                モデルの新規作成時のみ指定する
                すでにモデルが読み込まれている場合はこの値は無視される
            trainable:
                転移学習を行うときに特徴検出部分を再学習するかどうか
                すでにモデルが読み込まれている場合はこの値は無視される

        Returns:
            学習を行ったモデルの情報
        """
        train_ds, val_ds = self.create_dataset(dataset_path, batch_size, normalize=self.get_normalize_flag(model_type))

        class_image_num, class_indices = self.count_image_from_dataset(train_ds)
        val_class_image_num, val_class_indices = self.count_image_from_dataset(val_ds)

        if self.model is None:
            if model_type == define.ModelType.unknown:
                nlib3.print_error_log("モデルを新規作成する場合はモデルタイプを指定してください")
                return None
            self.model_data = {}
            self.model_data[define.AiDataKey.model] = model_type    # モデル作成時のみモデルタイプを登録する
            self.model_data[define.AiDataKey.trainable] = trainable
            self.model = self.create_model(model_type, len(class_indices), trainable)

        timetaken = tf_callback.TimeCallback()
        history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[timetaken])
        self.model.save_weights(MODEL_DIR / self.model_name / MODEL_FILE)
        self.model_data[define.AiDataKey.version] = self.MODEL_DATA_VERSION
        self.model_data[define.AiDataKey.ai_type] = self.ai_type
        self.model_data[define.AiDataKey.class_num] = len(class_indices)
        self.model_data[define.AiDataKey.class_indices] = class_indices
        self.model_data[define.AiDataKey.train_image_num] = class_image_num
        self.model_data[define.AiDataKey.test_image_num] = val_class_image_num

        for k, v in history.history.items():
            if k in self.model_data:
                self.model_data[k] += v
            else:
                self.model_data[k] = v

        nlib3.save_json(MODEL_DIR / self.model_name / f"{MODEL_FILE}.json", self.model_data)
        self.show_history()
        self.show_model_test(dataset_path, max_loop_num=5)
        return self.model_data.copy()

    def show_history(self) -> None:
        """モデルの学習履歴をグラフで表示する
        """
        fig = plt.figure(figsize=(6.4, 4.8 * 2))
        fig.suptitle("Learning history")
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(self.model_data[define.AiDataKey.accuracy])
        ax.plot(self.model_data[define.AiDataKey.val_accuracy])
        ax.set_title("Model accuracy")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Epoch")
        ax.legend(["Train", "Test"], loc="upper left")

        ax = fig.add_subplot(2, 1, 2)
        ax.plot(self.model_data[define.AiDataKey.loss])
        ax.plot(self.model_data[define.AiDataKey.val_loss])
        ax.set_title("Model loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        # ax.set_ylim(0, 0.3)
        ax.legend(["Train", "Test"], loc="upper left")
        plt.show()
        return

    def load_model(self) -> None:
        """学習済みニューラルネットワークの重みを読み込む
        """
        if self.model is None:
            self.model_data = nlib3.load_json(MODEL_DIR / self.model_name / f"{MODEL_FILE}.json")
            self.model = self.create_model(self.model_data[define.AiDataKey.model], self.model_data[define.AiDataKey.class_num], self.model_data[define.AiDataKey.trainable])
            self.model.load_weights(MODEL_DIR / self.model_name / MODEL_FILE)
        else:
            nlib3.print_error_log("既に初期化されています")
        return

    @abc.abstractmethod
    @model_required
    def inference():
        """画像のファイルパスを指定して推論する"""

    @abc.abstractmethod
    @model_required
    def show_model_test():
        """テストデータの推論結果を表示する"""
