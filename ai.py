import enum
import os
import pathlib
import random

os.environ['PATH'] += ";" + str(pathlib.Path(pathlib.Path(__file__).parent, "dll"))			# 環境変数に一時的に dll のパスを追加する

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

import image_manager
import lib
import tf_callback

MODEL_DIR = "./model"

class ModelType(str, enum.Enum):
	unknown = "unknown"
	vgg16_512 = "vgg16_512"

class DataKey(str, enum.Enum):
	version = "version"
	model = "model"
	class_num = "class_num"
	class_indices = "class_indices"
	accuracy = "accuracy"
	val_accuracy = "val_accuracy"
	loss = "loss"
	val_loss = "val_loss"

class ImageClassificationAi():
	MODEL_DATA_VERSION = 2

	def __init__(self, model_name: str):
		self.model = None
		self.model_data = None
		self.model_name = model_name
		self.img_height = 224
		self.img_width = 224
		return

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
		return None

	def create_model_vgg16(self, num_classes):
		# model = tf.keras.Sequential([
		# 	tf.keras.layers.Rescaling(1./255),
		# 	tf.keras.layers.Conv2D(32, 3, activation="relu"),
		# 	tf.keras.layers.MaxPooling2D(),
		# 	tf.keras.layers.Conv2D(32, 3, activation="relu"),
		# 	tf.keras.layers.MaxPooling2D(),
		# 	tf.keras.layers.Conv2D(32, 3, activation="relu"),
		# 	tf.keras.layers.MaxPooling2D(),
		# 	tf.keras.layers.Flatten(),
		# 	tf.keras.layers.Dense(128, activation="relu"),
		# 	tf.keras.layers.Dense(num_classes)
		# ])
		# vgg16_model = tf.keras.applications.vgg16.VGG16(include_top=False)

		# model = tf.keras.Sequential([
		# 	*vgg16_model.layers,
		# 	tf.keras.layers.Flatten(input_shape=vgg16_model.output_shape[1:]),
		# 	tf.keras.layers.Dense(256, activation="relu"),
		# 	tf.keras.layers.Dropout(0.5),
		# 	tf.keras.layers.Dense(num_classes, activation="softmax"),
		# ])

		# model.compile(
		# 	optimizer="adam",
		# 	loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
		# 	metrics=["accuracy"])



		# # `include_top=False`として既存モデルの出力層を消す。
		# vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=(224, 224, 3))

		# # モデルを編集する。
		# model = tf.keras.Sequential(vgg16.layers)

		# # 全19層のうち15層目までは再学習しないようにパラメータを固定する。
		# for layer in model.layers[:15]:
		# 	layer.trainable = False
		# # 出力層の部分を追加
		# model.add(tf.keras.layers.Flatten())
		# model.add(tf.keras.layers.Dense(256, activation="relu"))
		# model.add(tf.keras.layers.Dropout(0.5))
		# model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))



			# `include_top=False`として既存モデルの出力層を消す。
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
		model.summary()
		# 最適化アルゴリズムをSGD（確率的勾配降下法）とし最適化の学習率と修正幅を指定してコンパイルする。
		model.compile(
			loss="categorical_crossentropy",
			optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
			metrics=["accuracy"]
		)
		return model

	# 画像を読み込んでテンソルに変換する
	def preprocess_image(self, img_path):
		image = Image.open(img_path)
		image = image.convert("RGB")
		image = image.resize((self.img_height, self.img_width))
		data = np.asarray(image)
		x = []
		x.append(data)
		x = np.array(x)
		return x

		# img_raw = tf.io.read_file(str(img_path))
		# image = tf.image.decode_image(img_raw, channels=3)
		# image = tf.image.resize(image, [img_height, img_width])
		# image /= 255.0							# normalize to [0,1] range
		# image = tf.expand_dims(image, 0)		# 次元を一つ増やしてバッチ化する
		# return image

	def train_model(self, dataset_path, epochs: int = 6, model_type: ModelType = ModelType.unknown):
		batch_size = 32
		
		params = {
			# "rescale": 1./255,
			"horizontal_flip": True,			# 左右を反転する
			"rotation_range": 20,				# 度数法で最大変化時の角度を指定
			"channel_shift_range": 15,			# 各画素値の値を加算・減算します。パラメータとしては変化の最大量を指定します。
			"height_shift_range": 0.03,			# 中心位置を相対的にずらす ( 元画像の高さ X 値 の範囲内で垂直方向にずらす )
			"width_shift_range": 0.03,
			"validation_split": 0.1,			# 全体に対するテストデータの割合
		}
		generator = tf.keras.preprocessing.image.ImageDataGenerator(**params)
		train_ds = generator.flow_from_directory(
			dataset_path,
			target_size = (self.img_height, self.img_width),
			batch_size = batch_size,
			class_mode="categorical",
			subset = "training")
		val_ds = generator.flow_from_directory(
			dataset_path,
			target_size = (self.img_height, self.img_width),
			batch_size = batch_size,
			class_mode="categorical",
			subset = "validation")

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

		for k, v in history.history.items():
			if k in self.model_data:
				self.model_data[k] += v
			else:
				self.model_data[k] = v

		lib.save_json(pathlib.Path(MODEL_DIR, self.model_name + ".json"), self.model_data)
		self.show_history()
		self.check_model_sample(dataset_path)
		return

	# モデルの学習履歴をグラフで表示する
	def show_history(self, separate = True):
		if separate:
			plt.plot(self.model_data[DataKey.accuracy])
			plt.plot(self.model_data[DataKey.val_accuracy])
			plt.title("Model accuracy")
			plt.ylabel("Accuracy")
			plt.xlabel("Epoch")
			plt.legend(["Train", "Test"], loc="upper left")
			plt.show()

			plt.plot(self.model_data[DataKey.loss])
			plt.plot(self.model_data[DataKey.val_loss])
			plt.title("Model loss")
			plt.ylabel("Loss")
			plt.xlabel("Epoch")
			plt.legend(["Train", "Test"], loc="upper left")
			plt.show()
		else:
			plt.plot(self.model_data[DataKey.accuracy])
			plt.plot(self.model_data[DataKey.val_accuracy])
			plt.plot(self.model_data[DataKey.loss])
			plt.plot(self.model_data[DataKey.val_loss])
			plt.title("Model train history")
			plt.ylabel("Accuracy & loss")
			plt.xlabel("Epoch")
			plt.legend(["Train accuracy", "Test accuracy", "Train loss", "Test loss"], loc="upper left")
			plt.show()
		return

	def load_model(self):
		if self.model is None:
			self.model_data = lib.load_json(pathlib.Path(MODEL_DIR, self.model_name + ".json"))
			self.model = self.create_model(self.model_data[DataKey.model], self.model_data[DataKey.class_num])
			self.model.load_weights(pathlib.Path(MODEL_DIR, self.model_name))
		else:
			lib.print_error_log("既に初期化されています")
		return

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
				result = self.model(self.preprocess_image(image_path))[0]
				ax = fig.add_subplot(3, 8, i + 1)
				ax.imshow(np.asarray(img))
				ax = fig.add_subplot(3, 8, i + 2)
				color = "blue"
				if pathlib.Path(image_path.parent).name != class_name_list[list(result).index(max(result))]:
					color = "red"
				ax.bar(np.array(range(len(class_name_list))), result, tick_label=class_name_list, color=color)
			plt.show()
