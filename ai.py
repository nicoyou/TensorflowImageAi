import enum
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

import image_manager
import lib

MODEL_DIR = "./model"

class ModelType(str, enum.Enum):
	unknown = "unknown"
	vgg16_512 = "vgg16_512"

class DataKey(str, enum.Enum):
	model = "model"
	class_num = "class_num"
	class_indices = "class_indices"

class ImageClassificationAi():
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

	def create_model(self, model_type: ModelType, num_classes: int):
		match model_type:
			case ModelType.vgg16_512:
				return self.create_model_vgg16(num_classes)
		return None

	# 画像分類モデルを作成する
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
			optimizer=tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.1),
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

	def train_model(self, dataset_path, model_type: ModelType = ModelType.unknown):
		batch_size = 32
		train_ds = tf.keras.utils.image_dataset_from_directory(
			dataset_path,
			validation_split=0.15,
			subset="training",
			seed=123,
			image_size=(self.img_height, self.img_width),
			batch_size=batch_size,
			label_mode="categorical"				# 転移学習の場合はこれを指定しないとクラッシュする
		)
		val_ds = tf.keras.utils.image_dataset_from_directory(
			dataset_path,
			validation_split=0.15,
			subset="validation",
			seed=123,
			image_size=(self.img_height, self.img_width),
			batch_size=batch_size,
			label_mode="categorical"
		)
		class_names = train_ds.class_names
		if self.model is None:
			if model_type == ModelType.unknown:
				lib.print_error_log("モデルを新規作成する場合はモデルタイプを指定してください")
				return None
			self.model_data = {}
			self.model_data[DataKey.model] = model_type			# モデル作成時のみモデルタイプを上書きする
			self.model = self.create_model(model_type, len(class_names))
		# plt.figure(figsize=(10, 10))
		# for images, labels in train_ds.take(1):
		# 	for i in range(9):
		# 		ax = plt.subplot(3, 3, i + 1)
		# 		plt.imshow(images[i].numpy().astype("uint8"))
		# 		plt.title(train_ds.class_names[labels[i]])
		# 		plt.axis("off")

		# plt.show()

		# normalization_layer = tf.keras.layers.Rescaling(1./255)
		# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
		# image_batch, labels_batch = next(iter(normalized_ds))
		# first_image = image_batch[0]
		# # Notice the pixel values are now in `[0,1]`.
		# print(np.min(first_image), np.max(first_image))

		AUTOTUNE = tf.data.AUTOTUNE
		# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
		# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

		history = self.model.fit(train_ds, validation_data=val_ds, epochs=3)
		self.model.save_weights(pathlib.Path(MODEL_DIR, self.model_name))
		self.model_data[DataKey.class_num] = len(class_names)
		self.model_data[DataKey.class_indices] = train_ds.class_names

		lib.save_json(pathlib.Path(MODEL_DIR, self.model_name + ".json"), self.model_data)
		self.check_model_sample(dataset_path)
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
			fig = plt.figure(figsize=(19, 10))
			for i in range(0, 12*2, 2):
				image_path = random.choice(images)
				img = Image.open(image_path)
				result = self.model(self.preprocess_image(image_path))[0]
				ax = fig.add_subplot(3, 8, i + 1)
				ax.imshow(np.asarray(img))
				ax = fig.add_subplot(3, 8, i + 2)
				color = "blue"
				if pathlib.Path(image_path.parent).name != self.model_data[DataKey.class_indices][list(result).index(max(result))]:
					color = "red"
				ax.bar(np.array(range(len(self.model_data[DataKey.class_indices]))), result, tick_label=self.model_data[DataKey.class_indices], color=color)
			plt.show()
