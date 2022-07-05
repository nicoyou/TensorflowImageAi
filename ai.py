import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

import lib
import image_manager

MODEL_DIR = "./model"
img_height = 224
img_width = 224

# 画像分類モデルを作成する
def create_model_vgg16(num_classes):
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
	vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=(img_height, img_width, 3))

	top_model =  tf.keras.Sequential()
	top_model.add(tf.keras.layers.Flatten(input_shape=vgg16.output_shape[1:]))
	top_model.add(tf.keras.layers.Dense(256, activation="relu"))
	top_model.add(tf.keras.layers.Dropout(0.5))
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
		optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
		metrics=["accuracy"]
	)
	return model

# 画像を読み込んでテンソルに変換する
def preprocess_image(img_path):
	image = Image.open(img_path)
	image = image.convert("RGB")
	image = image.resize((img_height, img_width))
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

def train_model(dataset_path, model_name):
	batch_size = 32
	train_ds = tf.keras.utils.image_dataset_from_directory(
		dataset_path,
		validation_split=0.15,
		subset="training",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size,
		label_mode="categorical"				# 転移学習の場合はこれを指定しないとクラッシュする
	)
	val_ds = tf.keras.utils.image_dataset_from_directory(
		dataset_path,
		validation_split=0.15,
		subset="validation",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size,
		label_mode="categorical"
	)
	class_names = train_ds.class_names
	model = create_model_vgg16(len(class_names))

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

	history = model.fit(train_ds, validation_data=val_ds, epochs=3)
	model.save_weights(pathlib.Path(MODEL_DIR, model_name))
	model_data = {
		"class_num": len(class_names),
		"class_name": train_ds.class_names,
	}

	lib.save_json(pathlib.Path(MODEL_DIR, model_name + ".json"), model_data)
	check_model_sample(model, model_data, dataset_path)
	return model

def load_model(model_name):
	model_data = load_model_data(model_name)
	model = create_model_vgg16(model_data["class_num"])
	model.load_weights(pathlib.Path(MODEL_DIR, model_name))
	return model

def load_model_data(model_name):
	return lib.load_json(pathlib.Path(MODEL_DIR, model_name + ".json"))

def check_model_sample(model, model_data, dataset_path, loop_num = 5):
	random.seed(0)
	images = image_manager.get_image_path_from_dir(dataset_path)
	for row in range(loop_num):
		fig = plt.figure(figsize=(19, 10))
		for i in range(0, 12*2, 2):
			image_path = random.choice(images)
			img = Image.open(image_path)
			result = model(preprocess_image(image_path))[0]
			ax = fig.add_subplot(3, 8, i + 1)
			ax.imshow(np.asarray(img))
			ax = fig.add_subplot(3, 8, i + 2)
			color = "blue"
			if pathlib.Path(image_path.parent).name != model_data["class_name"][list(result).index(max(result))]:
				color = "red"
			ax.bar(np.array(range(len(model_data["class_name"]))), result, tick_label=model_data["class_name"], color=color)
		plt.show()
