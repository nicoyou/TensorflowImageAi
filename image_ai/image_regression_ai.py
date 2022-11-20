from . import ai

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nlib3
import pandas
import resnet_rs
import tensorflow as tf

from . import define


class ImageRegressionAi(ai.Ai):
    """画像の回帰分析AI"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(define.AiType.regression, *args, **kwargs)
        return

    def compile_model(self, model: Any, learning_rate: float = 0.0002):
        """モデルを回帰問題に最適なパラメータでコンパイルする

        Args:
            model: 未コンパイルのモデル
            learning_rate: 学習率

        Returns:
            コンパイル後のモデル
        """
        model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=[self.accuracy])
        return model

    def create_model(self, model_type: define.ModelType, num_classes: int, trainable: bool = False) -> Any:
        """画像の回帰分析モデルを作成する

        Args:
            model_type: モデルの種類
            num_classes: 分類するクラスの数 ( 回帰分析モデルでは使用しない )
            trainable: 転移学習時に特徴量抽出部を再学習するかどうか

        Returns:
            tensorflow のモデル
        """
        match model_type:
            case define.ModelType.resnet_rs152_256_regr:
                return self.create_model_resnet_rs_256(trainable)
            case define.ModelType.resnet_rs152_512x2_regr:
                return self.create_model_resnet_rs_512x2(trainable)
            case define.ModelType.efficient_net_v2_b0_regr:
                return self.create_model_efficient_net_v2_b0(trainable)
            case define.ModelType.efficient_net_v2_s_regr:
                return self.create_model_efficient_net_v2_s(trainable)
        return None

    @staticmethod
    def accuracy(y_true: Any, y_pred: Any) -> Any:
        """回帰問題における正答率に使用するスコアを計算する

        Args:
            y_true: 正しい値
            y_pred: 推論された値

        Returns:
            0 ～ 1 のスコア
        """
        pred_error = abs(y_true - y_pred)
        result = tf.divide(
            tf.reduce_sum(tf.cast(pred_error < 0.1, tf.float32)) * 0.5 + tf.reduce_sum(tf.cast(pred_error < 0.2, tf.float32)) * 0.5, # 誤差が 0.1 以下なら 1, 0.2 以下なら 0.5 とする
            tf.cast(len(y_pred), tf.float32))
        return result

    def create_model_resnet_rs_256(self, trainable: bool) -> Any:
        """ResNet_RSの転移学習モデルを作成する

        Args:
            num_classes: 分類するクラスの数
            trainable: 特徴量抽出部を再学習するかどうか

        Returns:
            tensorflow のモデル
        """
        resnet = resnet_rs.ResNetRS152(include_top=False, input_shape=define.DEFAULT_INPUT_SHAPE, weights="imagenet-i224")

        x = tf.keras.layers.Flatten(input_shape=resnet.output_shape[1:])(resnet.output)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        output = tf.keras.layers.Dense(1, activation="linear")(x)

        model = tf.keras.models.Model(inputs=resnet.input, outputs=output)

        if not trainable:
            for layer in model.layers[:779]:
                layer.trainable = False

        return self.compile_model(model)

    def create_model_resnet_rs_512x2(self, trainable: bool) -> Any:
        """ResNet_RSの転移学習モデルを作成する

        Args:
            trainable: 特徴量抽出部を再学習するかどうか

        Returns:
            tensorflow のモデル
        """
        resnet = resnet_rs.ResNetRS152(include_top=False, input_shape=define.DEFAULT_INPUT_SHAPE, weights="imagenet-i224")

        top_model = tf.keras.Sequential()
        top_model.add(tf.keras.layers.Flatten(input_shape=resnet.output_shape[1:]))
        top_model.add(tf.keras.layers.Dense(512, activation="relu"))
        top_model.add(tf.keras.layers.Dense(512, activation="relu"))
        top_model.add(tf.keras.layers.Dropout(0.25))
        top_model.add(tf.keras.layers.Dense(1, activation="linear"))

        model = tf.keras.models.Model(inputs=resnet.input, outputs=top_model(resnet.output))

        if not trainable:
            for layer in model.layers[:779]:
                layer.trainable = False

        return self.compile_model(model)

    def create_model_efficient_net_v2_b0(self, trainable: bool) -> Any:
        """EfficientNetV2B0の転移学習モデルを作成する

        Args:
            num_classes: 分類するクラスの数
            trainable: 特徴量抽出部を再学習するかどうか

        Returns:
            tensorflow のモデル
        """
        if not trainable:
            nlib3.print_error_log("EfficientNetV2 モデルでは trainable に False を指定できません")
        model = tf.keras.applications.EfficientNetV2B0(weights=None, classes=1, classifier_activation="linear")
        return self.compile_model(model, 0.001)

    def create_model_efficient_net_v2_s(self, trainable: bool) -> Any:
        """EfficientNetV2Sの転移学習モデルを作成する

        Args:
            num_classes: 分類するクラスの数
            trainable: 特徴量抽出部を再学習するかどうか

        Returns:
            tensorflow のモデル
        """
        if not trainable:
            nlib3.print_error_log("EfficientNetV2 モデルでは trainable に False を指定できません")
        model = tf.keras.applications.EfficientNetV2S(weights=None, classes=1, classifier_activation="linear")
        return self.compile_model(model, 0.001)

    def create_dataset(self, data_csv_path: str, batch_size: int, normalize: bool = False) -> tuple:
        """訓練用のデータセットを読み込む

        Args:
            data_csv_path: データセットの情報が格納された csv ファイルのパス
            batch_size: バッチサイズ
            normalize: 画像を前処理で 0 ～ 1 の範囲に正規化するかどうか

        Returns:
            train_ds: 訓練用のデータセット
            val_ds: テスト用のデータセット
        """
        generator = self.create_generator(normalize)
        df = pandas.read_csv(data_csv_path)
        df = df.dropna(subset=[self.y_col_name])                    # 空の行を取り除く
        df = df.sample(frac=1, random_state=0)                      # ランダムに並び変える
        train_ds = generator.flow_from_dataframe(
            df,
            directory=Path(data_csv_path).parent,                   # csv ファイルが保存されていたディレクトリを画像ファイルの親ディレクトリにする
            y_col=self.y_col_name,
            target_size=(self.image_size.y, self.image_size.x),
            batch_size=batch_size,
            seed=define.RANDOM_SEED,
            class_mode="raw",
            subset="training",
            validate_filenames=False,                               # パスチェックを行わない
        )
        val_ds = generator.flow_from_dataframe(
            df,
            directory=Path(data_csv_path).parent,
            y_col=self.y_col_name,
            target_size=(self.image_size.y, self.image_size.x),
            batch_size=batch_size,
            seed=define.RANDOM_SEED,
            class_mode="raw",
            subset="validation",
            validate_filenames=False,                               # パスチェックを行わない
        )
        return train_ds, val_ds                                     # [[[img*batch], [class*batch]], ...] の形式

    def count_image_from_dataset(self, dataset: Any) -> tuple:
        """データセットに含まれる画像の数を取得する

        Args:
            dataset: 画像数を計測するデータセット

        Returns:
            class_image_num: それぞれの値の画像数が格納されたリスト
            class_indices: データセットのそれぞれの値からクラスインデックスへのマッピングを含む辞書
        """
        if len(dataset) <= 0:
            nlib3.print_error_log("空のデータセットが渡されました")
        progbar = tf.keras.utils.Progbar(len(dataset))
        image_num = {}

        for i, row in enumerate(dataset):
            for value in row[1]:
                if value in image_num:
                    image_num[value] += 1
                else:
                    image_num[value] = 1

            progbar.update(i + 1)
            if i == len(dataset) - 1:   # 無限にループするため、最後まで取得したら終了する
                break
        dataset.reset()
        class_indices = {row: i for i, row in enumerate(list(image_num.keys()))}
        return list(image_num.values()), class_indices

    def init_model_type(self, model_type: define.ModelType) -> None:
        """モデルの種類に応じてパラメータを初期化する

        Args:
            model_type: モデルの種類
        """
        match model_type:
            case define.ModelType.efficient_net_v2_b0_regr:
                self.need_image_normalization = False
            case define.ModelType.efficient_net_v2_s_regr:
                self.need_image_normalization = False
                self.image_size.set(384, 384)
        return

    @ai.model_required
    def predict(self, image: str | Path | tf.Tensor) -> float:
        """画像を指定して推論する

        Args:
            image: 画像のファイルパスか、画像のテンソル

        Returns:
            推論結果
        """
        if isinstance(image, (str, Path)):
            image = self.preprocess_image(image, self.need_image_normalization)
        result = self.model(image)
        return float(result[0])

    @ai.model_required
    def show_model_test(self, data_csv_path: str, max_loop_num: int = 0, use_val_ds: bool = True) -> None:
        """テストデータの推論結果を表示する

        Args:
            data_csv_path: データセットの情報が格納された csv ファイルのパス
            max_loop_num: 結果を表示する最大回数 ( 1 回につき複数枚の画像が表示される )
            use_val_ds: データセットから訓練用の画像を使用するかどうか ( False でテスト用データを使用する )
        """
        train_ds, test_ds = self.create_dataset(data_csv_path, 12, normalize=self.need_image_normalization)
        if not use_val_ds:
            test_ds = train_ds

        for i, row in enumerate(test_ds):
            fig = plt.figure(figsize=(16, 9))
            for j in range(len(row[0])):
                result = self.model(tf.expand_dims(row[0][j], 0))[0]
                ax = fig.add_subplot(3, 8, j * 2 + 1)
                if self.need_image_normalization:
                    ax.imshow(row[0][j])
                else:
                    ax.imshow(tf.cast(row[0][j], tf.int32))
                ax = fig.add_subplot(3, 8, j * 2 + 2)
                color = "blue"
                if abs(row[1][j] - result[0]) > 0.17:
                    color = "orange"
                if abs(row[1][j] - result[0]) > 0.35:
                    color = "red"
                ax.bar([0, 1], [result[0], row[1][j]], tick_label=["ai", "true"], color=color)
                ax.set_ylim(0, 1.1)
            plt.show()
            if i == len(test_ds) - 1 or i == max_loop_num - 1:
                break
        return
