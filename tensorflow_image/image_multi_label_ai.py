from . import ai

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nlib3
import numpy as np
import pandas
import tensorflow as tf
from keras import backend

from . import define


class ImageMultiLabelAi(ai.Ai):
    """多ラベル分類AI"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(define.AiType.multi_label, *args, **kwargs)
        self.set_y_col_name("labels")
        return

    def compile_model(self, model: Any, learning_rate: float = 0.0002):
        """モデルを多ラベル問題に最適なパラメータでコンパイルする

        Args:
            model: 未コンパイルのモデル
            learning_rate: 学習率

        Returns:
            コンパイル後のモデル
        """
        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=[self.accuracy])
        return model

    def create_model(self, model_type: define.ModelType, num_classes: int, trainable: bool = False) -> Any | None:
        """多ラベル分類モデルを作成する

        Args:
            model_type: モデルの種類
            num_classes: 分類するラベルの数
            trainable: 転移学習時に特徴量抽出部を再学習するかどうか

        Returns:
            tensorflow のモデル
        """
        match model_type:
            case define.ModelType.resnet_rs152_256_multi_label:
                return self.create_model_resnet_rs_256(num_classes, trainable)
            case define.ModelType.resnet_rs152_512x2_multi_label:
                return self.create_model_resnet_rs_512x2(num_classes, trainable)
            case define.ModelType.efficient_net_v2_b0_multi_label:
                return self.create_model_efficient_net_v2_b0(num_classes, trainable)
            case define.ModelType.efficient_net_v2_s_multi_label:
                return self.create_model_efficient_net_v2_s(num_classes, trainable)
        return None

    @staticmethod
    def accuracy(y_true: Any, y_pred: Any) -> Any:
        """多ラベル問題における正答率に使用するスコアを計算する

        Args:
            y_true: 正しい値
            y_pred: 推論された値

        Returns:
            0 ～ 1 のスコア
        """
        pred = backend.cast(backend.greater_equal(y_pred, 0.5), tf.float32)
        flag = backend.cast(backend.equal(y_true, pred), tf.float32)
        return backend.mean(flag, axis=-1)

    def create_model_resnet_rs_256(self, num_classes: int, trainable: bool) -> Any:
        """ResNet_RSの転移学習モデルを作成する

        Args:
            num_classes: 分類するクラスの数
            trainable: 特徴量抽出部を再学習するかどうか

        Returns:
            tensorflow のモデル
        """
        import resnet_rs
        resnet = resnet_rs.ResNetRS152(include_top=False, input_shape=define.DEFAULT_INPUT_SHAPE, weights="imagenet-i224")

        x = tf.keras.layers.Flatten(input_shape=resnet.output_shape[1:])(resnet.output)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        output = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

        model = tf.keras.models.Model(inputs=resnet.input, outputs=output)

        if not trainable:
            for layer in model.layers[:779]:
                layer.trainable = False

        return self.model_compile(model)

    def create_model_resnet_rs_512x2(self, num_classes: int, trainable: bool) -> Any:
        """ResNet_RSの転移学習モデルを作成する

        Args:
            num_classes: 分類するラベルの数
            trainable: 特徴量抽出部を再学習するかどうか

        Returns:
            tensorflow のモデル
        """
        import resnet_rs
        resnet = resnet_rs.ResNetRS152(include_top=False, input_shape=define.DEFAULT_INPUT_SHAPE, weights="imagenet-i224")

        top_model = tf.keras.Sequential()
        top_model.add(tf.keras.layers.Flatten(input_shape=resnet.output_shape[1:]))
        top_model.add(tf.keras.layers.Dense(512, activation="relu"))
        top_model.add(tf.keras.layers.Dense(512, activation="relu"))
        top_model.add(tf.keras.layers.Dropout(0.25))
        top_model.add(tf.keras.layers.Dense(num_classes, activation="sigmoid"))

        model = tf.keras.models.Model(inputs=resnet.input, outputs=top_model(resnet.output))

        if not trainable:
            for layer in model.layers[:779]:
                layer.trainable = False

        return self.model_compile(model)

    def create_model_efficient_net_v2_b0(self, num_classes: int, trainable: bool) -> Any:
        """EfficientNetV2B0の転移学習モデルを作成する

        Args:
            num_classes: 分類するクラスの数
            trainable: 特徴量抽出部を再学習するかどうか

        Returns:
            tensorflow のモデル
        """
        if not trainable:
            nlib3.print_error_log("EfficientNetV2 モデルでは trainable に False を指定できません")
        model = tf.keras.applications.EfficientNetV2B0(weights=None, classes=num_classes, classifier_activation="sigmoid")
        return self.compile_model(model, 0.001)

    def create_model_efficient_net_v2_s(self, num_classes: int, trainable: bool) -> Any:
        """EfficientNetV2Sの転移学習モデルを作成する

        Args:
            num_classes: 分類するクラスの数
            trainable: 特徴量抽出部を再学習するかどうか

        Returns:
            tensorflow のモデル
        """
        if not trainable:
            nlib3.print_error_log("EfficientNetV2 モデルでは trainable に False を指定できません")
        model = tf.keras.applications.EfficientNetV2S(weights=None, classes=num_classes, classifier_activation="sigmoid")
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
        generator_val = self.create_generator(normalize, True)

        df = pandas.read_csv(data_csv_path)
        df = df.dropna(subset=[self.y_col_name])                    # 空の行を取り除く
        df[self.y_col_name] = df[self.y_col_name].str.split(",")    # 複数のラベルを格納している列は文字列からリストに変換する
        df = df.sample(frac=1, random_state=0)                      # ランダムに並び変える
        train_ds = generator.flow_from_dataframe(
            df,
            directory=str(Path(data_csv_path).parent),              # csv ファイルが保存されていたディレクトリを画像ファイルの親ディレクトリにする
            y_col=self.y_col_name,
            target_size=(self.image_size.y, self.image_size.x),
            batch_size=batch_size,
            seed=define.RANDOM_SEED,
            class_mode="categorical",
            subset="training",
            validate_filenames=False,                               # パスチェックを行わない
        )
        val_ds = generator_val.flow_from_dataframe(
            df,
            directory=str(Path(data_csv_path).parent),
            y_col=self.y_col_name,
            target_size=(self.image_size.y, self.image_size.x),
            batch_size=batch_size,
            seed=define.RANDOM_SEED,
            class_mode="categorical",
            subset="validation",
            validate_filenames=False,                               # パスチェックを行わない
        )
        return train_ds, val_ds                                     # [[[img*batch], [class*batch]], ...] の形式

    def count_image_from_dataset(self, dataset: Any) -> tuple:
        """データセットに含まれるラベルごとの画像の数を取得する

        Args:
            dataset: 画像数を計測するデータセット

        Returns:
            class_image_num: それぞれのラベルの画像数が格納されたリスト
            class_indices: データセットのラベル名からラベルインデックスへのマッピングを含む辞書
        """
        progbar = tf.keras.utils.Progbar(len(dataset))
        class_image_num = []
        for i in range(len(dataset.class_indices)):     # 各ラベルの読み込み枚数を 0 で初期化して、カウント用のキーを生成する ( 3 ラベル中の 1 番目なら[1, -1, -1] )
            class_image_num.append([0, [1 if i == row else -1 for row in range(len(dataset.class_indices))]])

        for i, row in enumerate(dataset):
            for image_num in class_image_num:                                                       # 各ラベルのデータ数を計測する
                image_num[0] += np.count_nonzero([np.any(x) for x in (row[1] == image_num[1])])     # numpyでキーが一致するものを any 条件でカウントする
            progbar.update(i + 1)
            if i == len(dataset) - 1:                                                               # 無限にループするため、最後まで取得したら終了する
                break
        class_image_num = [row for row, label in class_image_num]                                   # 不要になったラベルのキーを破棄する
        dataset.reset()
        return class_image_num, dataset.class_indices

    def init_model_type(self, model_type: define.ModelType) -> None:
        """モデルの種類に応じてパラメータを初期化する

        Args:
            model_type: モデルの種類
        """
        match model_type:
            case define.ModelType.efficient_net_v2_b0_multi_label:
                self.need_image_normalization = False
            case define.ModelType.efficient_net_v2_s_multi_label:
                self.need_image_normalization = False
                self.image_size.set(384, 384)
        return

    @ai.model_required
    def predict(self, image: str | Path | tf.Tensor) -> list:
        """画像を指定して推論する

        Args:
            image: 画像のファイルパスか、画像のテンソル

        Returns:
            各ラベルの確立を格納したリスト
        """
        if isinstance(image, (str, Path)):
            image = self.preprocess_image(image, self.need_image_normalization)
        result = self.model(image)
        return [float(row) for row in result[0]]

    @ai.model_required
    def result_to_labelname(self, result: list | tuple, border: float = 0.8) -> str:
        """推論結果のリストをラベル名のリストに変換する

        Args:
            result: predict 関数で得た推論結果
            border: ラベルを有効だとみなす最低値

        Returns:
            付与されたラベル名のリスト
        """
        label_name_list = list(self.model_data[define.AiDataKey.class_indices].keys())
        return [label_name_list[i] for i, row in enumerate(result) if row > border]

    @ai.model_required
    def result_to_label_dict(self, result: list | tuple) -> str:
        """推論結果のリストをラベル名をキーとした辞書に変換する

        Args:
            result: predict 関数で得た推論結果

        Returns:
            辞書に変換された推論結果
        """
        label_name_list = list(self.model_data[define.AiDataKey.class_indices].keys())
        return {label_name_list[i]: row for i, row in enumerate(result)}

    @ai.model_required
    def show_model_test(self, data_csv_path: str, max_loop_num: int = 0, use_val_ds: bool = True) -> None:
        """テストデータの推論結果を表示する

        Args:
            data_csv_path: データセットの情報が格納された csv ファイルのパス
            max_loop_num: 結果を表示する最大回数 ( 1 回につき複数枚の画像が表示される )
            use_val_ds: データセットから訓練用の画像を使用するかどうか ( False でテスト用データを使用する )
        """
        train_ds, test_ds = self.create_dataset(data_csv_path, 12, normalize=self.need_image_normalization)     # バッチサイズを表示する画像数と同じにする
        if not use_val_ds:
            test_ds = train_ds

        for i, row in enumerate(test_ds):
            fig = plt.figure(figsize=(16, 9))
            for j in range(len(row[0])):                                    # 最大12の画像数
                result = self.predict(tf.expand_dims(row[0][j], 0))
                result = self.result_to_label_dict(result)
                result_true = self.result_to_label_dict(row[1][j])
                result = [[k, v] for k, v in result.items()]
                result = sorted(result, reverse=True, key=lambda x: x[1])   # 確率が高いものから順に表示する
                ax = fig.add_subplot(3, 8, j * 2 + 1)
                if self.need_image_normalization:
                    ax.imshow(row[0][j])
                else:
                    ax.imshow(tf.cast(row[0][j], tf.int32))
                ax = fig.add_subplot(3, 8, j * 2 + 2)
                for iy, (k, v) in enumerate(result[:10]):
                    if v < 0.01:                                            # 確率が 1 % 以下のものは表示しない
                        break
                    color = "black"
                    if v > 0.7 and result_true[k] == 1:                     # 教師データにも存在するラベルを検出できた場合
                        color = "green"
                    elif v > 0.7 and result_true[k] == 0:                   # 教師データにないラベルを検出した場合
                        color = "red"
                    elif v <= 0.7 and result_true[k] == 1:                  # 教師データに存在するラベルを検出できなかった場合
                        color = "orange"
                    ax.text(0, 1 - (iy + 1) * 0.1, f" {k}: {v:.2f}", color=color)
            plt.show()
            if i == len(test_ds) - 1 or i == max_loop_num - 1:              # 表示した回数がバッチ数を超えたら終了する
                break
        return
