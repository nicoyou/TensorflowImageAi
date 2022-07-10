import enum
import os
import pathlib
import random

os.environ['PATH'] += ";" + str(pathlib.Path(pathlib.Path(__file__).parent, "dll"))			# 環境変数に一時的に dll のパスを追加する

import matplotlib.pyplot as plt
import numpy as np
import resnet_rs
import tensorflow as tf
from PIL import Image

import image_manager
import lib
import tf_callback

MODEL_DIR = "./model"

class ModelType(str, enum.Enum):
	unknown = "unknown"
	vgg16_512 = "vgg16_512"
	resnet_rs152_512x2 = "resnet_rs152_512x2"

class DataKey(str, enum.Enum):
	version = "version"
	model = "model"
	class_num = "class_num"
	class_indices = "class_indices"
	train_image_num = "train_image_num"
	accuracy = "accuracy"
	val_accuracy = "val_accuracy"
	loss = "loss"
	val_loss = "val_loss"

class ImageClassificationAi():
	MODEL_DATA_VERSION = 3

	def __init__(self, model_name: str):
		self.model = None
		self.model_data = None
		self.model_name = model_name
		self.img_height = 224
		self.img_width = 224
		return

	# モデルが定義されていなければ実行できない関数のデコレーター
	def model_required(func):
		def wrapper(self, *args, **kwargs):
			if self.model is None:
				lib.print_error_log("モデルが初期化されていません")
				return None
			return func(self, *args, **kwargs)
		return wrapper

	# 画像分類モデルを作成する
	def create_model(self, model_type: ModelType, num_classes: int):
		match model_type:
			case ModelType.vgg16_512:
				return self.create_model_vgg16(num_classes)
			case ModelType.resnet_rs152_512x2:
				return self.create_model_resnet_rs(num_classes)
		return None

	# ResNet_RSモデルを作成する
	def create_model_resnet_rs(self, num_classes):
		resnet = resnet_rs.ResNetRS152(include_top=False, input_shape=(self.img_height, self.img_width, 3), weights="imagenet-i224")

		top_model = tf.keras.Sequential()
		top_model.add(tf.keras.layers.Flatten(input_shape=resnet.output_shape[1:]))
		top_model.add(tf.keras.layers.Dense(512, activation="relu"))
		top_model.add(tf.keras.layers.Dense(512, activation="relu"))
		top_model.add(tf.keras.layers.Dropout(0.25))
		top_model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

		model = tf.keras.models.Model(
			inputs=resnet.input,
			outputs=top_model(resnet.output)
		)

		for layer in model.layers[:762]:
			layer.trainable = False

		model.compile(
			loss="categorical_crossentropy",
			optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
			metrics=["accuracy"]
		)
		return model

	# vgg16から転移学習するためのmodelを作成する
	def create_model_vgg16(self, num_classes):
		vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=(self.img_height, self.img_width, 3))

		top_model =  tf.keras.Sequential()
		top_model.add(tf.keras.layers.Flatten(input_shape=vgg16.output_shape[1:]))
		top_model.add(tf.keras.layers.Dense(512, activation="relu"))
		top_model.add(tf.keras.layers.Dropout(0.25))
		top_model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

		model = tf.keras.models.Model(
			inputs=vgg16.input,
			outputs=top_model(vgg16.output)
		)

		for layer in model.layers[:15]:
			layer.trainable = False

		# 最適化アルゴリズムをSGD（確率的勾配降下法）とし最適化の学習率と修正幅を指定してコンパイルする。
		model.compile(
			loss="categorical_crossentropy",
			optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
			metrics=["accuracy"]
		)
		return model

	# 画像を読み込んでテンソルに変換する
	def preprocess_image(self, img_path, normalize = False):
		img_raw = tf.io.read_file(str(img_path))
		image = tf.image.decode_image(img_raw, channels=3)
		image = tf.image.resize(image, (self.img_height, self.img_width))
		if normalize:
			image /= 255.0						# normalize to [0,1] range
		image = tf.expand_dims(image, 0)		# 次元を一つ増やしてバッチ化する
		return image

	# ノーマライズが必要かどうかを取得する
	def get_normalize_flag(self, model_type: ModelType = ModelType.unknown) -> bool:
		normalize = True
		no_normalize_model = [ModelType.vgg16_512]
		if (self.model_data is not None and self.model_data[DataKey.model] in no_normalize_model) or model_type in no_normalize_model:
			normalize = False
		return normalize

	# データセットの前処理を行うジェネレーターを作成する
	def create_generator(self, normalize):
		params = {
			"horizontal_flip": True,			# 左右を反転する
			"rotation_range": 20,				# 度数法で最大変化時の角度を指定
			"channel_shift_range": 15,			# 各画素値の値を加算・減算します。パラメータとしては変化の最大量を指定します。
			"height_shift_range": 0.03,			# 中心位置を相対的にずらす ( 元画像の高さ X 値 の範囲内で垂直方向にずらす )
			"width_shift_range": 0.03,
			"validation_split": 0.1,			# 全体に対するテストデータの割合
		}
		if normalize:
			params["rescale"] = 1. / 255
		return tf.keras.preprocessing.image.ImageDataGenerator(**params)

	# 訓練用のデータセットを生成する
	def create_dataset(self, dataset_path, batch_size, normalize = False):
		generator = self.create_generator(normalize)
		train_ds = generator.flow_from_directory(
			dataset_path,
			target_size = (self.img_height, self.img_width),
			batch_size = batch_size,
			seed=54,
			class_mode="categorical",
			subset = "training")
		val_ds = generator.flow_from_directory(
			dataset_path,
			target_size = (self.img_height, self.img_width),
			batch_size = batch_size,
			seed=54,
			class_mode="categorical",
			subset = "validation")
		return train_ds, val_ds		# [[[img*batch], [class*batch]], ...] の形式

	# ディープラーニングを開始する
	def train_model(self, dataset_path: str, epochs: int = 6, model_type: ModelType = ModelType.unknown) -> dict:
		train_ds, val_ds = self.create_dataset(dataset_path, 32, normalize=self.get_normalize_flag(model_type))

		progbar = tf.keras.utils.Progbar(len(train_ds))
		class_image_num = []
		for i in range(len(train_ds.class_indices)):			# 各クラスの読み込み枚数を 0 で初期化して、カウント用のキーを生成する ( ３クラス中の１番目なら [1, 0, 0])
			class_image_num.append([0, [1 if i == row else 0 for row in range(len(train_ds.class_indices))]])

		for i, row in enumerate(train_ds):
			for image_num in class_image_num:					# 各クラスのデータ数を計測する
				image_num[0] += np.count_nonzero([np.all(x) for x in (row[1] == image_num[1])])		# numpyでキーが一致するものをカウントする
			progbar.update(i + 1)
			if i == len(train_ds) - 1:							# 無限にループするため、最後まで取得したら終了する
				break
		class_image_num = [row for row, label in class_image_num]		# 不要になったラベルのキーを破棄する
		train_ds.reset()

		if self.model is None:
			if model_type == ModelType.unknown:
				lib.print_error_log("モデルを新規作成する場合はモデルタイプを指定してください")
				return None
			self.model_data = {}
			self.model_data[DataKey.model] = model_type			# モデル作成時のみモデルタイプを登録する
			self.model = self.create_model(model_type, len(train_ds.class_indices))

		timetaken = tf_callback.TimeCallback()
		history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[timetaken])
		self.model.save_weights(pathlib.Path(MODEL_DIR, self.model_name))
		self.model_data[DataKey.version] = self.MODEL_DATA_VERSION
		self.model_data[DataKey.class_num] = len(train_ds.class_indices)
		self.model_data[DataKey.class_indices] = train_ds.class_indices
		self.model_data[DataKey.train_image_num] = class_image_num

		for k, v in history.history.items():
			if k in self.model_data:
				self.model_data[k] += v
			else:
				self.model_data[k] = v

		lib.save_json(pathlib.Path(MODEL_DIR, self.model_name + ".json"), self.model_data)
		self.show_history()
		self.check_model_sample(dataset_path)
		return self.model_data.copy()

	# モデルの学習履歴をグラフで表示する
	def show_history(self):
		fig = plt.figure(figsize=(6.4, 4.8 * 2))
		fig.suptitle("Learning history")
		ax = fig.add_subplot(2, 1, 1)
		ax.plot(self.model_data[DataKey.accuracy])
		ax.plot(self.model_data[DataKey.val_accuracy])
		ax.set_title("Model accuracy")
		ax.set_ylabel("Accuracy")
		ax.set_xlabel("Epoch")
		ax.legend(["Train", "Test"], loc="upper left")

		ax = fig.add_subplot(2, 1, 2)
		ax.plot(self.model_data[DataKey.loss])
		ax.plot(self.model_data[DataKey.val_loss])
		ax.set_title("Model loss")
		ax.set_ylabel("Loss")
		ax.set_xlabel("Epoch")
		ax.legend(["Train", "Test"], loc="upper left")
		plt.show()
		return

	# 学習済みのニューラルネットワークを読み込む
	def load_model(self):
		if self.model is None:
			self.model_data = lib.load_json(pathlib.Path(MODEL_DIR, self.model_name + ".json"))
			self.model = self.create_model(self.model_data[DataKey.model], self.model_data[DataKey.class_num])
			self.model.load_weights(pathlib.Path(MODEL_DIR, self.model_name))
		else:
			lib.print_error_log("既に初期化されています")
		return

	# ランダムな画像でモデルの推論結果を表示する
	@model_required
	def check_model_sample(self, dataset_path, loop_num = 5):
		random.seed(0)
		images = image_manager.get_image_path_from_dir(dataset_path)
		for row in range(loop_num):
			class_name_list = list(self.model_data[DataKey.class_indices].keys())
			fig = plt.figure(figsize=(19, 10))
			for i in range(0, 12*2, 2):
				image_path = random.choice(images)
				img = Image.open(image_path)
				result = self.model(self.preprocess_image(image_path, self.get_normalize_flag()))[0]
				ax = fig.add_subplot(3, 8, i + 1)
				ax.imshow(np.asarray(img))
				ax = fig.add_subplot(3, 8, i + 2)
				color = "blue"
				if pathlib.Path(image_path.parent).name != class_name_list[list(result).index(max(result))]:
					color = "red"
				ax.bar(np.array(range(len(class_name_list))), result, tick_label=class_name_list, color=color)
			plt.show()
		return
