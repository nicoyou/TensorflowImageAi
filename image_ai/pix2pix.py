from . import ai
import datetime
import glob
import os
import statistics
import time
from pathlib import Path
from typing import Any, Final

import matplotlib
import nlib3
import numpy as np
import PIL
import tensorflow as tf
from matplotlib import pyplot as plt

from . import define, make_pix2pix_image


class PixToPixModel():
    OUTPUT_CHANNELS: Final[int] = 3

    def downsample(self, filters: int, size: int, apply_batchnorm: bool = True) -> Any:
        """ダウンサンプラー
        """
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding="same", kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())
        return result

    def upsample(self, filters: int, size: int, apply_dropout: bool = False) -> Any:
        """アップサンプラー
        """
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding="same", kernel_initializer=initializer, use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())
        return result

    def get_generator(self) -> Any:
        """ジェネレーターのモデルを取得する

        Returns:
            ジェネレーターのモデル
        """
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])

        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            self.downsample(128, 4),                        # (batch_size, 64, 64, 128)
            self.downsample(256, 4),                        # (batch_size, 32, 32, 256)
            self.downsample(512, 4),                        # (batch_size, 16, 16, 512)
            self.downsample(512, 4),                        # (batch_size, 8, 8, 512)
            self.downsample(512, 4),                        # (batch_size, 4, 4, 512)
            self.downsample(512, 4),                        # (batch_size, 2, 2, 512)
            self.downsample(512, 4),                        # (batch_size, 1, 1, 512)
        ]
        up_stack = [
            self.upsample(512, 4, apply_dropout=True),      # (batch_size, 2, 2, 1024)
            self.upsample(512, 4, apply_dropout=True),      # (batch_size, 4, 4, 1024)
            self.upsample(512, 4, apply_dropout=True),      # (batch_size, 8, 8, 1024)
            self.upsample(512, 4),                          # (batch_size, 16, 16, 1024)
            self.upsample(256, 4),                          # (batch_size, 32, 32, 512)
            self.upsample(128, 4),                          # (batch_size, 64, 64, 256)
            self.upsample(64, 4),                           # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4, strides=2, padding="same", kernel_initializer=initializer, activation="tanh") # (batch_size, 256, 256, 3)

        x = inputs

        # モデルによるダウンサンプリング
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # アップサンプリングとスキップコネクションの確立
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def get_discriminator(self) -> Any:
        """弁別器のモデルを取得する

        Returns:
            弁別器のモデル
        """
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 3], name="input_image")
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name="target_image")

        x = tf.keras.layers.concatenate([inp, tar])     # (batch_size, 256, 256, channels*2)

        down1 = self.downsample(64, 4, False)(x)    # (batch_size, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)      # (batch_size, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)      # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)                                                              # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)     # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)     # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)   # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)


class PixToPix():
    MODEL_DIR = define.CURRENT_DIR / "models"
    LOG_DIR = define.CURRENT_DIR / "logs"
    CHECKPOINT_FILE_NAME = Path("ckpt")
    MODEL_FILE_NAME = Path("model")
    JSON_FILE_NAME = Path("model.json")

    BUFFER_SIZE = 512                       # データセットを構成している画像の枚数
    BATCH_SIZE = 1                          # 元の pix2pix 実験では、バッチ サイズ 1 のほうが U-Net でより良い結果が出る
    IMAGE_SIZE = nlib3.Vector2(256, 256)    # 画像サイズ

    LAMBDA = 100

    def __init__(self, ai_name: str) -> None:
        """pix2pixのモデルを作成する

        Args:
            ai_name: AIの名前 ( 保存、読み込み用 )
        """
        self.ai_name = ai_name
        self.model_data = {}
        self.model_data[define.AiDataKey.version] = 2
        self.model_data[define.AiDataKey.ai_type] = define.AiType.gan
        self.model_data[define.AiDataKey.model] = define.ModelType.pix2pix
        self.model_data[define.AiDataKey.trainable] = True

        self.model_data[define.AiDataKey.train_image_num] = 0
        self.model_data[define.AiDataKey.test_image_num] = 0

        self.model_data[define.GanDataKey.gen_total_loss] = []
        self.model_data[define.GanDataKey.gen_gan_loss] = []
        self.model_data[define.GanDataKey.gen_l1_loss] = []
        self.model_data[define.GanDataKey.disc_loss] = []
        self.model_data[define.GanDataKey.time] = []

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        pix2pix_model = PixToPixModel()
        self.generator = pix2pix_model.get_generator()
        # ジェネレーターのモデル構造を図に出力する
        #tf.keras.utils.plot_model(self.generator, show_shapes=True, dpi=64)

        self.discriminator = pix2pix_model.get_discriminator()
        # 分別機を図に出力する
        #tf.keras.utils.plot_model(self.discriminator, show_shapes=True, dpi=64)

        # オプティマイザー
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # チェックポイントセーバー
        self.checkpoint_prefix = str(self.MODEL_DIR / self.ai_name / self.CHECKPOINT_FILE_NAME)
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

        self.summary_writer = tf.summary.create_file_writer(str(self.LOG_DIR / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        return

    def load_one_image(self, file_path: str) -> tf.Tensor:
        """pix2pix モデルで使用する形式で画像を読み込む

        Args:
            file_path: 読み込む画像のファイルパス

        Returns:
            読み込んだ画像のテンソル
        """
        pil_img = PIL.Image.open(file_path)
        pil_img = pil_img.convert("RGB")
        pil_img = make_pix2pix_image.expand_to_rect_pad(pil_img)
        pil_img = pil_img.resize(self.IMAGE_SIZE)
        tf_image = tf.keras.preprocessing.image.img_to_array(pil_img)
        tf_image = (tf_image / 127.5) - 1
        return tf.expand_dims(tf_image, 0)

    def predict(self, input_img: tf.Tensor) -> tf.Tensor:
        """現在保持しているモデルで画像の推論を行う

        Args:
            input_img: 入力画像データ

        Returns:
            変換後の画像データ
        """
        return self.generator(input_img, training=True)

    def show_predict(self, file_path: str, out_path: str = None) -> None:
        """指定された画像の推論結果を表示する

        Args:
            file_path: 入力画像のファイルパス
            out_path: 入力画像と出力画像の比較画像を出力するファイルパス
        """
        input_img = self.load_one_image(file_path)
        prediction = self.predict(input_img)
        plt.figure(figsize=(12, 6))

        display_list = [input_img[0], prediction[0]]
        title = ["Input Image", "Predicted Image"]

        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.title(title[i])
            # プロットする [0, 1] 範囲のピクセル値を取得する
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis("off")

        if out_path is not None:
            plt.savefig(out_path)
            return

        plt.show()
        plt.clf()       # 図形のクリア
        plt.close()     # windowを閉じる
        return

    def predict_image(self, input_image_path: str, out_image_path: str) -> None:
        """指定された画像をAIで変換して保存する

        Args:
            input_image: 入力画像のファイルパス
            out_image: 変換後の画像を保存するファイルパス
        """
        pil_image = PIL.Image.open(input_image_path)        # アスペクト比を計算するために読み込む
        tf_image = self.load_one_image(input_image_path)    # 推論用に読み込む
        tf_image = self.predict(tf_image)
        tf_image = (tf_image[0] + 1) * 127.5

        result_image = PIL.Image.fromarray(tf_image.numpy().astype(np.uint8))

        original_size = nlib3.Vector2(pil_image.size[0], pil_image.size[1])
        out_size = ((original_size / original_size.max()) * self.IMAGE_SIZE).floor()
        out_pad_size = ((self.IMAGE_SIZE - out_size) / 2).floor()

        if original_size.x != original_size.y:
            result_image = result_image.crop((*out_pad_size, *(self.IMAGE_SIZE - out_pad_size)))

        result_image.save(out_image_path)
        return result_image

    def load(self, image_file: str) -> tuple:
        """pix2pix用の教師データを読み込んで、２つの画像テンソルに分解する

        Args:
            image_file: 読み込む画像のファイルパス

        Returns:
            入力画像と目的画像のテンソルを格納したタプル
            (入力画像, 目的画像)
        """
        # 画像ファイルを読み取って uint8 テンソルにデコードする
        image = tf.io.read_file(image_file)
        image = tf.io.decode_jpeg(image)

        # 各画像テンソルを２つのテンソルに分割する
        w = tf.shape(image)[1]
        w = w // 2
        input_image = image[:, w:, :]
        real_image = image[:, :w, :]

        # 両方の画像を float32 テンソルに変換する
        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)
        return input_image, real_image

    def resize(self, input_image: tf.Tensor, real_image: tf.Tensor, height: int, width: int) -> tuple:
        """教師データの画像をリサイズする

        Args:
            input_image: 入力画像のテンソル
            real_image: 目的画像のテンソル
            height: リサイズする画像の高さ
            width: リサイズ後の幅

        Returns:
            入力画像と目的画像のテンソルを格納したタプル
            (入力画像, 目的画像)
        """
        input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return input_image, real_image

    def random_crop(self, input_image: tf.Tensor, real_image: tf.Tensor) -> tuple:
        """教師データをランダムな位置でトリミングする

        Args:
            input_image: 入力画像のテンソル
            real_image: 目的画像のテンソル

        Returns:
            入力画像と目的画像のテンソルを格納したタプル
            (入力画像, 目的画像)
        """
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2, self.IMAGE_SIZE.y, self.IMAGE_SIZE.x, 3])

        return cropped_image[0], cropped_image[1]

    def normalize(self, input_image: tf.Tensor, real_image: tf.Tensor) -> tuple:
        """画像を [-1, 1] の範囲に正規化する

        Args:
            input_image: 入力画像のテンソル
            real_image: 目的画像のテンソル

        Returns:
            入力画像と目的画像のテンソルを格納したタプル
            (入力画像, 目的画像)
        """
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image

    @tf.function()
    def random_jitter(self, input_image: tf.Tensor, real_image: tf.Tensor) -> tuple:
        """訓練用データの前処理を行う

        Args:
            input_image: 入力画像のテンソル
            real_image: 目的画像のテンソル

        Returns:
            入力画像と目的画像のテンソルを格納したタプル
            (入力画像, 目的画像)
        """
        input_image, real_image = self.resize(input_image, real_image, 286, 286)
        input_image, real_image = self.random_crop(input_image, real_image)     # ランダムにトリミングして元のサイズに戻す

        if tf.random.uniform(()) > 0.5:     # ランダムにミラーリングする
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def show_image_sample(self, dataset: Any) -> None:
        """前処理された画像のサンプルを表示する

        Args:
            dataset: load_dataset() 関数で読み込んだデータセット
        """
        loop_num = len(dataset)
        if loop_num > 5:
            loop_num = 5
        plt.figure(figsize=(6, 6))
        for inp, re in dataset.take(loop_num):
            for i in range(4):
                rj_inp, rj_re = self.random_jitter(inp[0], re[0])
                plt.subplot(2, 2, i + 1)
                plt.imshow(rj_inp * 0.5 + 0.5)
                plt.axis(False)
            plt.show()
        return

    def load_image_train(self, image_file: str) -> tuple:
        """訓練用の教師画像を読み込んで入力画像と目的画像に分割する
        また、画像にランダム水増しの前処理を追加する

        Args:
            image_file: 入力画像のファイルパス

        Returns:
            入力画像と目的画像のテンソルを格納したタプル
            (入力画像, 目的画像)
        """
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def load_image_test(self, image_file: str) -> tuple:
        """テスト用の教師画像を読み込んで入力画像と目的画像に分割する

        Args:
            image_file: テスト用画像のファイルパス

        Returns:
            入力画像と目的画像のテンソルを格納したタプル
            (入力画像, 目的画像)
        """
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.resize(input_image, real_image, self.IMAGE_SIZE.y, self.IMAGE_SIZE.x)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def generator_loss(self, disc_generated_output, gen_output, target):
        """ジェネレーターの損失計算を行う
        """
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # 平均絶対誤差
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """弁別器の損失計算を行う
        """
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def generate_images(self, test_input: tf.Tensor, tar: tf.Tensor, step: int | None = None) -> None:
        """入力画像、教師画像、出力画像を横に並べてに出力する

        Args:
            test_input: 入力画像のテンソル
            tar: 目的画像のテンソル
            step: 現在のステップ数 ( 指定することでモデルディレクトリ内に画像を出力する )
        """
        prediction = self.generator(test_input, training=True)
        plt.figure(figsize=(15, 6))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ["Input Image", "Ground Truth", "Predicted Image"]

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # プロットする [0, 1] 範囲のピクセル値を取得する
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis("off")

        os.makedirs("train_log_images", exist_ok=True)
        if step is not None:
            os.makedirs(str(self.MODEL_DIR / self.ai_name / "images"), exist_ok=True)
            plt.savefig(self.MODEL_DIR / self.ai_name / "images" / f"{step:07}.png")
        else:
            plt.show()
        plt.clf()       # 図形のクリア
        plt.close()     # windowを閉じる
        return

    def load_dataset(self, dataset_dir: str) -> tuple:
        """データセットが保存されているディレクトリを指定してデータセットを読み込む
        指定されたディレクトリ内の「train」ディレクトリと「test」ディレクトリからそれぞれの画像をすべて読み込む
        画像はすべてpix2pixの教師データ形式で拡張子は png のみ

        Args:
            dataset_dir: データセットが保存されているディレクトリパス

        Returns:
            入力画像と目的画像のデータセットを格納したタプル
            (入力画像のデータセット, 目的画像のデータセット)
        """
        train_dataset = tf.data.Dataset.list_files(str(Path(dataset_dir) / "train" / "*.png"))
        train_dataset = train_dataset.map(self.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.BUFFER_SIZE)
        train_dataset = train_dataset.batch(self.BATCH_SIZE)

        try:
            test_dataset = tf.data.Dataset.list_files(str(Path(dataset_dir) / "test" / "*.png"))
        except tf.errors.InvalidArgumentError:
            test_dataset = tf.data.Dataset.list_files(str(Path(dataset_dir) / "val" / "*.png"))
        test_dataset = test_dataset.map(self.load_image_test)
        test_dataset = test_dataset.batch(self.BATCH_SIZE)
        return train_dataset, test_dataset

    @tf.function
    def train_step(self, input_image: tf.Tensor, target: tf.Tensor, step: int) -> tuple:
        """１ステップ分の学習を行う

        Args:
            input_image: 入力画像のテンソル
            target: 教師データの出力画像テンソル
            step: 現在のステップ数

        Returns:
            それぞれの損失を格納したタプル
            (gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss)
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=step // 1000)
            tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step // 1000)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step // 1000)
            tf.summary.scalar("disc_loss", disc_loss, step=step // 1000)
        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

    def train_model(self, dataset_dir: str, steps: int) -> None:
        """モデルの学習を行う

        Args:
            dataset_dir: データセットが保存されているディレクトリパス
            steps: 実際に学習を行うステップ数
        """
        STEP_INTERVAL = 1000
        train_ds, test_ds = self.load_dataset(dataset_dir)
        self.model_data[define.AiDataKey.train_image_num] = len(train_ds)
        self.model_data[define.AiDataKey.test_image_num] = len(test_ds)

        example_input, example_target = next(iter(test_ds.take(1)))
        start_time = time.time()
        os.makedirs(self.MODEL_DIR / self.ai_name, exist_ok=True)
        matplotlib.use("Agg")   # メモリリーク対策

        gen_total_loss_list = []
        gen_gan_loss_list = []
        gen_l1_loss_list = []
        disc_loss_list = []

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if step % STEP_INTERVAL == 0:
                if step != 0:
                    self.model_data[define.GanDataKey.time].append(time.time() - start_time)
                start_time = time.time()
                progbar = tf.keras.utils.Progbar(STEP_INTERVAL)
                print(f"Step: {step // 1000}k")

            if (step < 100 and step % 10 == 0) or (step < 1000 and step % 100 == 0) or step % 1000 == 0:
                self.generate_images(example_input, example_target, step)

            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.train_step(input_image, target, step)

            gen_total_loss_list.append(float(gen_total_loss))
            gen_gan_loss_list.append(float(gen_gan_loss))
            gen_l1_loss_list.append(float(gen_l1_loss))
            disc_loss_list.append(float(disc_loss))
            if step % (STEP_INTERVAL // 10) == 0:
                self.model_data[define.GanDataKey.gen_total_loss].append(statistics.mean(gen_total_loss_list))
                self.model_data[define.GanDataKey.gen_gan_loss].append(statistics.mean(gen_gan_loss_list))
                self.model_data[define.GanDataKey.gen_l1_loss].append(statistics.mean(gen_l1_loss_list))
                self.model_data[define.GanDataKey.disc_loss].append(statistics.mean(disc_loss_list))
                gen_total_loss_list.clear()
                gen_gan_loss_list.clear()
                gen_l1_loss_list.clear()
                disc_loss_list.clear()

            # 訓練のステップ
            progbar.update(int(step) % STEP_INTERVAL + 1)

            # 10kステップごとにチェックポイントを保存する
            if (step + 1) % 10000 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                nlib3.save_json(self.MODEL_DIR / self.ai_name / self.JSON_FILE_NAME, self.model_data)

                # 古いチェックポイントを削除する
                ckpt_files = glob.glob(str(self.MODEL_DIR / self.ai_name / self.CHECKPOINT_FILE_NAME) + "-*")
                for i, row in enumerate(ckpt_files):
                    ckpt_files[i] = Path(row).stem
                    ckpt_files[i] = str(ckpt_files[i]).replace(str(self.CHECKPOINT_FILE_NAME) + "-", "")
                    ckpt_files[i] = int(ckpt_files[i])
                if max(ckpt_files) % 10 != 1:
                    os.remove(str(self.MODEL_DIR / self.ai_name / (str(self.CHECKPOINT_FILE_NAME) + f"-{max(ckpt_files) - 1}.data-00000-of-00001")))
                    os.remove(str(self.MODEL_DIR / self.ai_name / (str(self.CHECKPOINT_FILE_NAME) + f"-{max(ckpt_files) - 1}.index")))
        return

    def load_model(self) -> None:
        """モデルを読み込む ( チェックポイントからモデルの重みを復元する )
        """
        self.checkpoint.restore(tf.train.latest_checkpoint(str(self.MODEL_DIR / self.ai_name)))
        self.model_data = nlib3.load_json(self.MODEL_DIR / self.ai_name / self.JSON_FILE_NAME)
        return

    def show_test(self, dataset_dir: str, use_test_ds: bool = True) -> None:
        """指定されたデータセットの推論結果を一つずつ順番に表示する

        Args:
            dataset_dir: データセットが保存されているディレクトリパス
            use_test_ds: テスト用のデータを使用する
        """
        train_ds, test_ds = self.load_dataset(dataset_dir)
        if not use_test_ds:
            test_ds = train_ds
        for inp, tar in test_ds.take(len(test_ds)):
            self.generate_images(inp, tar)
        return

    def show_history(self) -> None:
        """モデルの学習履歴をグラフで表示する
        """
        fig = plt.figure(figsize=(6.4 * 2, 4.8 * 2))
        fig.suptitle("Learning history")
        ax = fig.add_subplot(2, 2, 1)
        ax.plot(self.model_data[define.GanDataKey.disc_loss])
        ax.set_title("Disc loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")

        ax = fig.add_subplot(2, 2, 2)
        ax.plot(self.model_data[define.GanDataKey.gen_gan_loss])
        ax.set_title("Gen gan loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")

        ax = fig.add_subplot(2, 2, 3)
        ax.plot(self.model_data[define.GanDataKey.gen_l1_loss])
        ax.set_title("Gen l1 loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")

        ax = fig.add_subplot(2, 2, 4)
        ax.plot(self.model_data[define.GanDataKey.gen_total_loss])
        ax.set_title("Gen total loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        plt.show()
        return
