from . import ai

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nlib3
import numpy as np
import pandas
import resnet_rs
import tensorflow as tf

from . import define


class ImageClassificationAi(ai.Ai):
    """多クラス分類AI"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(define.AiType.categorical, *args, **kwargs)
        return

    def compile_model(self, model: Any, learning_rate: float = 0.0002):
        """モデルを分類問題に最適なパラメータでコンパイルする

        Args:
            model: 未コンパイルのモデル
            learning_rate: 学習率

        Returns:
            コンパイル後のモデル
        """
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )
        return model

    def create_model(self, model_type: define.ModelType, num_classes: int, trainable: bool = False) -> Any | None:
        """画像分類モデルを作成する

        Args:
            model_type: モデルの種類
            num_classes: 分類するクラスの数
            trainable: 転移学習時に特徴量抽出部を再学習するかどうか

        Returns:
            tensorflow のモデル
        """
        match model_type:
            case define.ModelType.vgg16_512:
                return self.create_model_vgg16(num_classes, trainable)
            case define.ModelType.mobile_net_v2:
                return self.create_model_mobile_net_v2(num_classes, trainable)
            case define.ModelType.resnet_rs152_256:
                return self.create_model_resnet_rs_256(num_classes, trainable)
            case define.ModelType.resnet_rs152_512x2:
                return self.create_model_resnet_rs_512x2(num_classes, trainable)
            case define.ModelType.efficient_net_v2_b0:
                return self.create_model_efficient_net_v2_b0(num_classes, trainable)
            case define.ModelType.efficient_net_v2_s:
                return self.create_model_efficient_net_v2_s(num_classes, trainable)
        return None

    def create_model_vgg16(self, num_classes: int, trainable: bool) -> Any:
        """vgg16の転移学習モデルを作成する

        Args:
            num_classes: 分類するクラスの数
            trainable: 特徴量抽出部を再学習するかどうか

        Returns:
            tensorflow のモデル
        """
        vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=define.DEFAULT_INPUT_SHAPE)

        top_model = tf.keras.Sequential()
        top_model.add(tf.keras.layers.Flatten(input_shape=vgg16.output_shape[1:]))
        top_model.add(tf.keras.layers.Dense(512, activation="relu"))
        top_model.add(tf.keras.layers.Dropout(0.25))
        top_model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

        model = tf.keras.models.Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

        if not trainable:
            for layer in model.layers[:15]:
                layer.trainable = False

        return self.compile_model(model, 0.0001)

    def create_model_mobile_net_v2(self, num_classes: int, trainable: bool) -> Any:
        """MobileNetV2の転移学習モデルを作成する

        Args:
            num_classes: 分類するクラスの数
            trainable: 特徴量抽出部を再学習するかどうか

        Returns:
            tensorflow のモデル
        """
        mobile_net_v2 = tf.keras.applications.mobilenet_v2.MobileNetV2(classes=num_classes, weights=None)

        if not trainable:
            for layer in mobile_net_v2.layers[:154]:
                layer.trainable = False

        return self.compile_model(mobile_net_v2, 0.001)

    def create_model_resnet_rs_256(self, num_classes: int, trainable: bool) -> Any:
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

        output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=resnet.input, outputs=output)

        if not trainable:
            for layer in model.layers[:779]:
                layer.trainable = False

        return self.compile_model(model)

    def create_model_resnet_rs_512x2(self, num_classes: int, trainable: bool) -> Any:
        """ResNet_RSの転移学習モデルを作成する

        Args:
            num_classes: 分類するクラスの数
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
        top_model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

        model = tf.keras.models.Model(inputs=resnet.input, outputs=top_model(resnet.output))

        if not trainable:
            for layer in model.layers[:779]:
                layer.trainable = False

        return self.compile_model(model)

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
        model = tf.keras.applications.EfficientNetV2B0(weights=None, classes=num_classes)
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
        model = tf.keras.applications.EfficientNetV2S(weights=None, classes=num_classes)
        return self.compile_model(model, 0.001)

    def create_dataset(self, dataset_path: str, batch_size: int, normalize: bool = False) -> tuple:
        """訓練用のデータセットを読み込む

        Args:
            dataset_path: 教師データが保存されているディレクトリを指定する ( csvでも可能 )
            batch_size: バッチサイズ
            normalize: 画像を前処理で 0 ～ 1 の範囲に正規化するかどうか

        Returns:
            train_ds: 訓練用のデータセット
            val_ds: テスト用のデータセット
        """
        generator = self.create_generator(normalize)
        generator_val = self.create_generator(normalize, True)

        if Path(dataset_path).is_dir():
            train_ds = generator.flow_from_directory(
                dataset_path,
                target_size=(self.image_size.y, self.image_size.x),
                batch_size=batch_size,
                seed=define.RANDOM_SEED,
                class_mode="categorical",
                subset="training",
            )
            val_ds = generator_val.flow_from_directory(
                dataset_path,
                target_size=(self.image_size.y, self.image_size.x),
                batch_size=batch_size,
                seed=define.RANDOM_SEED,
                class_mode="categorical",
                subset="validation",
            )
        else:
            df = pandas.read_csv(dataset_path)
            df = df.dropna(subset=[self.y_col_name])                    # 空の行を取り除く
            df = df.sample(frac=1, random_state=0)                      # ランダムに並び変える
            train_ds = generator.flow_from_dataframe(
                df,
                directory=str(Path(dataset_path).parent),              # csv ファイルが保存されていたディレクトリを画像ファイルの親ディレクトリにする
                y_col=self.y_col_name,
                target_size=(self.image_size.y, self.image_size.x),
                batch_size=batch_size,
                seed=define.RANDOM_SEED,
                class_mode="categorical",
                subset="training",
                validate_filenames=False,                               # パスチェックを行わない
            )
            val_ds = generator.flow_from_dataframe(
                df,
                directory=str(Path(dataset_path).parent),
                y_col=self.y_col_name,
                target_size=(self.image_size.y, self.image_size.x),
                batch_size=batch_size,
                seed=define.RANDOM_SEED,
                class_mode="categorical",
                subset="validation",
                validate_filenames=False,                               # パスチェックを行わない
            )
        return train_ds, val_ds     # [[[img*batch], [class*batch]], ...] の形式

    def count_image_from_dataset(self, dataset: Any) -> tuple:
        """データセットに含まれるクラスごとの画像の数を取得する

        Args:
            dataset: 画像数を計測するデータセット

        Returns:
            class_image_num: それぞれの分類クラスの画像数が格納されたリスト
            class_indices: データセットのクラス名からクラスインデックスへのマッピングを含む辞書
        """
        progbar = tf.keras.utils.Progbar(len(dataset))
        class_image_num = []
        for i in range(len(dataset.class_indices)):     # 各クラスの読み込み枚数を 0 で初期化して、カウント用のキーを生成する ( 3 クラス中の 1 番目なら[1, 0, 0] )
            class_image_num.append([0, [1 if i == row else 0 for row in range(len(dataset.class_indices))]])

        for i, row in enumerate(dataset):
            for image_num in class_image_num:                                                       # 各クラスのデータ数を計測する
                image_num[0] += np.count_nonzero([np.all(x) for x in (row[1] == image_num[1])])     # numpyでキーが一致するものをカウントする
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
            case define.ModelType.vgg16_512:
                self.need_image_normalization = False
            case define.ModelType.efficient_net_v2_b0:
                self.need_image_normalization = False
            case define.ModelType.efficient_net_v2_s:
                self.need_image_normalization = False
                self.image_size.set(384, 384)
        return

    @ai.model_required
    def predict(self, image: str | Path | tf.Tensor) -> list:
        """画像を指定して推論する

        Args:
            image: 画像のファイルパスか、画像のテンソル

        Returns:
            各クラスの確立を格納したリスト
        """
        if isinstance(image, (str, Path)):
            image = self.preprocess_image(image, self.need_image_normalization)
        result = self.model(image)
        return [float(row) for row in result[0]]

    @ai.model_required
    def result_to_classname(self, result: list | tuple) -> str:
        """推論結果のリストをクラス名に変換する

        Args:
            result: predict 関数で得た推論結果

        Returns:
            クラス名
        """
        class_name_list = list(self.model_data[define.AiDataKey.class_indices].keys())
        return class_name_list[list(result).index(max(result))]

    @ai.model_required
    def show_model_test(self, dataset_path: str, max_loop_num: int = 0, use_val_ds: bool = True) -> None:
        """テストデータの推論結果を表示する

        Args:
            dataset_path: テストに使用するデータセットのディレクトリ
            max_loop_num: 結果を表示する最大回数 ( 1 回につき複数枚の画像が表示される )
            use_val_ds: データセットから訓練用の画像を使用するかどうか ( False でテスト用データを使用する )
        """
        train_ds, test_ds = self.create_dataset(dataset_path, 12, normalize=self.need_image_normalization)  # バッチサイズを表示する画像数と同じにする
        if not use_val_ds:
            test_ds = train_ds

        for i, row in enumerate(test_ds):
            class_name_list = list(self.model_data[define.AiDataKey.class_indices].keys())
            fig = plt.figure(figsize=(16, 9))
            for j in range(len(row[0])):                        # 最大12の画像数
                result = self.model(tf.expand_dims(row[0][j], 0))[0]
                ax = fig.add_subplot(3, 8, j * 2 + 1)
                if self.need_image_normalization:
                    ax.imshow(row[0][j])
                else:
                    ax.imshow(tf.cast(row[0][j], tf.int32))
                ax = fig.add_subplot(3, 8, j * 2 + 2)
                color = "blue"
                if list(row[1][j]).index(1) != list(result).index(max(result)):
                    color = "red"
                ax.bar(np.array(range(len(class_name_list))), result, tick_label=class_name_list, color=color)
            plt.show()
            if i == len(test_ds) - 1 or i == max_loop_num - 1:  # 表示した回数がバッチ数を超えたら終了する
                break
        return

    @ai.model_required
    def get_model_miss_list(self, dataset_path: str, use_val_ds: bool = True, print_result: bool = False) -> list:
        """テストデータで分類に失敗したデータリストを取得する

        Args:
            dataset_path: 推論に使用するデータセットのディレクトリ
            use_val_ds: データセットから訓練用の画像を使用するかどうか ( False でテスト用データを使用する )
            print_result: 間違えている推論結果をコンソールに表示する

        Returns:
            間違えた推論結果と本来のクラスを格納したタプルのリスト
            [(推論結果, 解答), (推論結果, 解答)...]
        """
        train_ds, test_ds = self.create_dataset(dataset_path, 8, normalize=self.need_image_normalization)
        result_list = []
        if not use_val_ds:
            test_ds = train_ds

        for i, row in enumerate(test_ds):
            for j in range(len(row[0])):    # 最大12の画像数
                result = self.model(tf.expand_dims(row[0][j], 0))[0]
                if list(row[1][j]).index(1) != list(result).index(max(result)):
                    result_list.append(([float(row) for row in result], list(row[1][j])))
                    if print_result:
                        result_class = self.result_to_classname(result)
                        true_class = self.result_to_classname(row[1][j])
                        print(f"{result_class} -> {true_class}")
            if i == len(test_ds) - 1:
                break
        return result_list
