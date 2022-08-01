import datetime
import enum
import os
import statistics
import time
from pathlib import Path

os.environ["PATH"] += ";" + str(Path(__file__).parent / "dll")			# 環境変数に一時的に dll のパスを追加する
os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={str(Path(__file__).parent / 'CUDA')}"

import nlib3
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib

import ai

matplotlib.use("Agg")			# メモリリーク対策

class GanDataKey(str, enum.Enum):
	gen_total_loss = "gen_total_loss"
	gen_gan_loss = "gen_gan_loss"
	gen_l1_loss = "gen_l1_loss"
	disc_loss = "disc_loss"

class PixToPixModel():
	OUTPUT_CHANNELS = 3

	# ダウンサンプラー
	def downsample(self, filters, size, apply_batchnorm=True):
		initializer = tf.random_normal_initializer(0., 0.02)

		result = tf.keras.Sequential()
		result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding="same", kernel_initializer=initializer, use_bias=False))

		if apply_batchnorm:
			result.add(tf.keras.layers.BatchNormalization())

		result.add(tf.keras.layers.LeakyReLU())

		return result

	# アップサンプラー
	def upsample(self, filters, size, apply_dropout=False):
		initializer = tf.random_normal_initializer(0., 0.02)

		result = tf.keras.Sequential()
		result.add(tf.keras.layers.Conv2DTranspose(
			filters, size, strides=2, padding="same", kernel_initializer=initializer, use_bias=False))
		result.add(tf.keras.layers.BatchNormalization())

		if apply_dropout:
			result.add(tf.keras.layers.Dropout(0.5))

		result.add(tf.keras.layers.ReLU())
		return result

	# ジェネレーター
	def get_generator(self):
		inputs = tf.keras.layers.Input(shape=[256, 256, 3])

		down_stack = [
			self.downsample(64, 4, apply_batchnorm=False),			# (batch_size, 128, 128, 64)
			self.downsample(128, 4),								# (batch_size, 64, 64, 128)
			self.downsample(256, 4),								# (batch_size, 32, 32, 256)
			self.downsample(512, 4),								# (batch_size, 16, 16, 512)
			self.downsample(512, 4),								# (batch_size, 8, 8, 512)
			self.downsample(512, 4),								# (batch_size, 4, 4, 512)
			self.downsample(512, 4),								# (batch_size, 2, 2, 512)
			self.downsample(512, 4),								# (batch_size, 1, 1, 512)
		]

		up_stack = [
			self.upsample(512, 4, apply_dropout=True),		# (batch_size, 2, 2, 1024)
			self.upsample(512, 4, apply_dropout=True),		# (batch_size, 4, 4, 1024)
			self.upsample(512, 4, apply_dropout=True),		# (batch_size, 8, 8, 1024)
			self.upsample(512, 4),							# (batch_size, 16, 16, 1024)
			self.upsample(256, 4),							# (batch_size, 32, 32, 512)
			self.upsample(128, 4),							# (batch_size, 64, 64, 256)
			self.upsample(64, 4),							# (batch_size, 128, 128, 128)
		]

		initializer = tf.random_normal_initializer(0., 0.02)
		last = tf.keras.layers.Conv2DTranspose(
			self.OUTPUT_CHANNELS, 4, strides=2, padding="same", kernel_initializer=initializer, activation="tanh")			# (batch_size, 256, 256, 3)

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

	# 弁別器
	def get_discriminator(self):
		initializer = tf.random_normal_initializer(0., 0.02)

		inp = tf.keras.layers.Input(shape=[256, 256, 3], name="input_image")
		tar = tf.keras.layers.Input(shape=[256, 256, 3], name="target_image")

		x = tf.keras.layers.concatenate([inp, tar])		# (batch_size, 256, 256, channels*2)

		down1 = self.downsample(64, 4, False)(x)		# (batch_size, 128, 128, 64)
		down2 = self.downsample(128, 4)(down1)			# (batch_size, 64, 64, 128)
		down3 = self.downsample(256, 4)(down2)			# (batch_size, 32, 32, 256)

		zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)					# (batch_size, 34, 34, 256)
		conv = tf.keras.layers.Conv2D(512, 4, strides=1,
			kernel_initializer=initializer, use_bias=False)(zero_pad1)		# (batch_size, 31, 31, 512)

		batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

		leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

		zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)				# (batch_size, 33, 33, 512)

		last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)		# (batch_size, 30, 30, 1)

		return tf.keras.Model(inputs=[inp, tar], outputs=last)

class PixToPix():
	MODEL_DIR = Path("./models")
	LOG_DIR = Path("./logs")
	MODEL_FILE_NAME = Path("model")
	JSON_FILE_NAME = Path("model.json")
	
	BUFFER_SIZE = 400				# データセットを構成している画像の枚数
	BATCH_SIZE = 1					# 元の pix2pix 実験では、バッチ サイズ 1 のほうが U-Net でより良い結果が出る
	IMG_WIDTH = 256					# 画像サイズ
	IMG_HEIGHT = 256

	LAMBDA = 100

	def __init__(self, ai_name) -> None:
		self.ai_name = ai_name
		self.model_data = {}
		self.model_data[ai.DataKey.version] = 1
		self.model_data[ai.DataKey.ai_type] = ai.AiType.gan
		self.model_data[ai.DataKey.model] = ai.ModelType.pix2pix
		self.model_data[ai.DataKey.trainable] = True

		self.model_data[GanDataKey.gen_total_loss] = []
		self.model_data[GanDataKey.gen_gan_loss] = []
		self.model_data[GanDataKey.gen_l1_loss] = []
		self.model_data[GanDataKey.disc_loss] = []

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
		self.checkpoint_prefix = str(self.MODEL_DIR / self.ai_name / self.MODEL_FILE_NAME)
		self.checkpoint = tf.train.Checkpoint(
			generator_optimizer=self.generator_optimizer,
			discriminator_optimizer=self.discriminator_optimizer,
			generator=self.generator,
			discriminator=self.discriminator)

		self.summary_writer = tf.summary.create_file_writer(str(self.LOG_DIR / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
		return

	# 画像ファイルを読み込んで、２つの画像テンソルに分解する
	def load(self, image_file):
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

	# 画像をリサイズする
	def resize(self, input_image, real_image, height, width):
		input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		return input_image, real_image

	# ランダムな位置でトリミングする
	def random_crop(self, input_image, real_image):
		stacked_image = tf.stack([input_image, real_image], axis=0)
		cropped_image = tf.image.random_crop(
			stacked_image, size=[2, self.IMG_HEIGHT, self.IMG_WIDTH, 3])

		return cropped_image[0], cropped_image[1]

	# 画像を [-1, 1] の範囲に正規化する
	def normalize(self, input_image, real_image):
		input_image = (input_image / 127.5) - 1
		real_image = (real_image / 127.5) - 1

		return input_image, real_image

	# 訓練用データの前処理を行う
	@tf.function()
	def random_jitter(self, input_image, real_image):
		# 286x286 にリサイズする
		input_image, real_image = self.resize(input_image, real_image, 286, 286)

		# ランダムにトリミングして 256x256 に戻す
		input_image, real_image = self.random_crop(input_image, real_image)

		if tf.random.uniform(()) > 0.5:
			# ランダムにミラーリングする
			input_image = tf.image.flip_left_right(input_image)
			real_image = tf.image.flip_left_right(real_image)

		return input_image, real_image

	# 前処理された出力の一部を表示する
	def show_image_sample(self):
		plt.figure(figsize=(6, 6))
		inp, re = test_dataset.take(1)
		for i in range(4):
			rj_inp, rj_re = self.random_jitter(inp, re)
			plt.subplot(2, 2, i + 1)
			plt.imshow(rj_inp / 255.0)
			plt.axis("off")
		plt.show()

	# 訓練用の画像を読み込む
	def load_image_train(self, image_file):
		input_image, real_image = self.load(image_file)
		input_image, real_image = self.random_jitter(input_image, real_image)
		input_image, real_image = self.normalize(input_image, real_image)

		return input_image, real_image

	# テスト用の画像を読み込む
	def load_image_test(self, image_file):
		input_image, real_image = self.load(image_file)
		input_image, real_image = self.resize(input_image, real_image, self.IMG_HEIGHT, self.IMG_WIDTH)
		input_image, real_image = self.normalize(input_image, real_image)

		return input_image, real_image

	# ジェネレーターの損失計算を行う
	def generator_loss(self, disc_generated_output, gen_output, target):
		gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

		# 平均絶対誤差
		l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
		total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)
		return total_gen_loss, gan_loss, l1_loss

	# 弁別器の損失計算を行う
	def discriminator_loss(self, disc_real_output, disc_generated_output):
		real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

		generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
		total_disc_loss = real_loss + generated_loss
		return total_disc_loss

	# 3つの画像を表示する
	def generate_images(self, model, test_input, tar, step):
		prediction = model(test_input, training=True)
		plt.figure(figsize=(15, 6))

		display_list = [test_input[0], tar[0], prediction[0]]
		title = ["Input Image", "Ground Truth", "Predicted Image"]

		for i in range(3):
			plt.subplot(1, 3, i+1)
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
		plt.clf()		# 図形のクリア
		plt.close()		# windowを閉じる
		return

	# データセットを読み込む
	def load_dataset(self, dataset_dir):
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

	# １ステップの学習を行う
	@tf.function
	def train_step(self, input_image, target, step):
		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			gen_output = self.generator(input_image, training=True)

			disc_real_output = self.discriminator([input_image, target], training=True)
			disc_generated_output = self.discriminator([input_image, gen_output], training=True)

			gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
			disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

		generator_gradients = gen_tape.gradient(gen_total_loss,
												self.generator.trainable_variables)
		discriminator_gradients = disc_tape.gradient(disc_loss,
												self.discriminator.trainable_variables)

		self.generator_optimizer.apply_gradients(zip(generator_gradients,
												self.generator.trainable_variables))
		self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
												self.discriminator.trainable_variables))

		with self.summary_writer.as_default():
			tf.summary.scalar("gen_total_loss", gen_total_loss, step=step // 1000)
			tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step // 1000)
			tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step // 1000)
			tf.summary.scalar("disc_loss", disc_loss, step=step // 1000)
		return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

	# 学習を行う
	def fit(self, train_ds, test_ds, steps):
		example_input, example_target = next(iter(test_ds.take(1)))
		start = time.time()
		os.makedirs(self.MODEL_DIR / self.ai_name, exist_ok=True)

		STEP_INTERVAL = 500

		gen_total_loss_list = []
		gen_gan_loss_list = []
		gen_l1_loss_list = []
		disc_loss_list = []

		for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
			if step % STEP_INTERVAL == 0:
				if step != 0:
					print(f"Time taken for {STEP_INTERVAL} steps: {time.time()-start:.2f} sec\n")

				start = time.time()
				print(f"Step: {step / 1000:.1f}k")

			if (step < 100 and step % 10 == 0) or (step < 1000 and step % 100 == 0) or step % 1000 == 0:
				self.generate_images(self.generator, example_input, example_target, step)

			gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.train_step(input_image, target, step)

			gen_total_loss_list.append(float(gen_total_loss))
			gen_gan_loss_list.append(float(gen_gan_loss))
			gen_l1_loss_list.append(float(gen_l1_loss))
			disc_loss_list.append(float(disc_loss))
			if step % (STEP_INTERVAL // 10) == 0:
				self.model_data["gen_total_loss"].append(statistics.mean(gen_total_loss_list))
				self.model_data["gen_gan_loss"].append(statistics.mean(gen_gan_loss_list))
				self.model_data["gen_l1_loss"].append(statistics.mean(gen_l1_loss_list))
				self.model_data["disc_loss"].append(statistics.mean(disc_loss_list))
				gen_total_loss_list.clear()
				gen_gan_loss_list.clear()
				gen_l1_loss_list.clear()
				disc_loss_list.clear()

			# 訓練のステップ
			if (step + 1) % 10 == 0:
				print(".", end="", flush=True)

			# 10kステップごとにチェックポイントを保存する
			if (step + 1) % 10000 == 0:
				self.checkpoint.save(file_prefix=self.checkpoint_prefix)
				nlib3.save_json(self.MODEL_DIR / self.ai_name / self.JSON_FILE_NAME, self.model_data)
		return

	# モデルを読み込む
	def load_model(self):
		self.checkpoint.restore(tf.train.latest_checkpoint(str(self.MODEL_DIR / self.ai_name)))
		self.model_data = nlib3.load_json(self.MODEL_DIR / self.ai_name / self.JSON_FILE_NAME)
		return

	# データセットの推論結果を表示する
	def show_test(self, dataset):
		for inp, tar in dataset.take(len(dataset)):
			self.generate_images(self.generator, inp, tar, step=None)
		return

	# モデルの学習履歴をグラフで表示する
	def show_history(self):
		fig = plt.figure(figsize=(6.4 * 2, 4.8 * 2))
		fig.suptitle("Learning history")
		ax = fig.add_subplot(2, 2, 1)
		ax.plot(self.model_data[GanDataKey.disc_loss])
		ax.set_title("Disc loss")
		ax.set_ylabel("Loss")
		ax.set_xlabel("Epoch")

		ax = fig.add_subplot(2, 2, 2)
		ax.plot(self.model_data[GanDataKey.gen_gan_loss])
		ax.set_title("Gen gan loss")
		ax.set_ylabel("Loss")
		ax.set_xlabel("Epoch")

		ax = fig.add_subplot(2, 2, 3)
		ax.plot(self.model_data[GanDataKey.gen_l1_loss])
		ax.set_title("Gen l1 loss")
		ax.set_ylabel("Loss")
		ax.set_xlabel("Epoch")

		ax = fig.add_subplot(2, 2, 4)
		ax.plot(self.model_data[GanDataKey.gen_total_loss])
		ax.set_title("Gen total loss")
		ax.set_ylabel("Loss")
		ax.set_xlabel("Epoch")
		plt.show()
		return

if __name__ == "__main__":
	p2p = PixToPix("pix2pix_naked_girl_256")
	train_dataset, test_dataset = p2p.load_dataset("dataset/out_p2p")
	p2p.load_model()
	p2p.show_history()
	p2p.show_test(test_dataset)
	# p2p.fit(train_dataset, test_dataset, steps=50000000)
