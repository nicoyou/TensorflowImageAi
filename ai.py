import abc
import enum
import os
from pathlib import Path

os.environ["PATH"] += ";" + str(Path(__file__).parent / "dll")			# 環境変数に一時的に dll のパスを追加する

import matplotlib.pyplot as plt
import nlib3
import numpy as np
import pandas
import resnet_rs
import tensorflow as tf

import define
import tf_callback

__version__ = "1.1.0"
MODEL_DIR = "./models"
MODEL_FILE = "model"
RANDOM_SEED = 54
	
# モデルが定義されていなければ実行できない関数のデコレーター
def model_required(func):
	def wrapper(self, *args, **kwargs):
		if self.model is None:
			nlib3.print_error_log("モデルが初期化されていません")
			return None
		return func(self, *args, **kwargs)
	return wrapper

class Ai(metaclass = abc.ABCMeta):
	MODEL_DATA_VERSION = 6

	def __init__(self, ai_type: define.AiType, model_name: str):
		self.model = None
		self.model_data = None
		self.model_name = model_name
		self.ai_type = ai_type
		self.img_height = 224
		self.img_width = 224
		return

	# 画像を読み込んでテンソルに変換する
	def preprocess_image(self, img_path, normalize = False):
		img_raw = tf.io.read_file(str(img_path))
		image = tf.image.decode_image(img_raw, channels=3)
		image = tf.image.resize(image, (self.img_height, self.img_width))
		if normalize:
			image /= 255.0						# normalize to [0,1] range
		image = tf.expand_dims(image, 0)		# 次元を一つ増やしてバッチ化する
		return image

	# pillowで読み込んだ画像をテンソルに変換する
	def pillow_image_to_tf_image(self, image, normalize = False):
		tf.keras.preprocessing.image.img_to_array(image)
		image = tf.image.resize(image, (self.img_height, self.img_width))
		if normalize:
			image /= 255.0						# normalize to [0,1] range
		image = tf.expand_dims(image, 0)		# 次元を一つ増やしてバッチ化する
		return image

	# ノーマライズが必要かどうかを取得する
	def get_normalize_flag(self, model_type: define.ModelType = define.ModelType.unknown) -> bool:
		normalize = True
		no_normalize_model = [define.ModelType.vgg16_512]
		if (self.model_data is not None and self.model_data[define.AiDataKey.model] in no_normalize_model) or model_type in no_normalize_model:
			normalize = False
		return normalize

	# データセットの前処理を行うジェネレーターを作成する
	def create_generator(self, normalize: bool):
		params = {
			"horizontal_flip": True,			# 左右を反転する
			"rotation_range": 20,				# 度数法で最大変化時の角度を指定
			"channel_shift_range": 15,			# 各画素値の値を加算・減算する ( 最大値を指定する )
			"height_shift_range": 0.03,			# 中心位置を相対的にずらす ( 元画像の高さ X 値 の範囲内で垂直方向にずらす )
			"width_shift_range": 0.03,
			"validation_split": 0.1,			# 全体に対するテストデータの割合
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

	# ディープラーニングを開始する
	def train_model(self, dataset_path: str, epochs: int = 6, batch_size: int = 32, model_type: define.ModelType = define.ModelType.unknown, trainable: bool = False) -> dict:
		train_ds, val_ds = self.create_dataset(dataset_path, batch_size, normalize=self.get_normalize_flag(model_type))

		class_image_num, class_indices = self.count_image_from_dataset(train_ds)
		val_class_image_num, val_class_indices = self.count_image_from_dataset(val_ds)

		if self.model is None:
			if model_type == define.ModelType.unknown:
				nlib3.print_error_log("モデルを新規作成する場合はモデルタイプを指定してください")
				return None
			self.model_data = {}
			self.model_data[define.AiDataKey.model] = model_type			# モデル作成時のみモデルタイプを登録する
			self.model = self.create_model(model_type, len(class_indices), trainable)

		timetaken = tf_callback.TimeCallback()
		history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[timetaken])
		self.model.save_weights(Path(MODEL_DIR, self.model_name, MODEL_FILE))
		self.model_data[define.AiDataKey.version] = self.MODEL_DATA_VERSION
		self.model_data[define.AiDataKey.ai_type] = self.ai_type
		self.model_data[define.AiDataKey.trainable] = trainable
		self.model_data[define.AiDataKey.class_num] = len(class_indices)
		self.model_data[define.AiDataKey.class_indices] = class_indices
		self.model_data[define.AiDataKey.train_image_num] = class_image_num
		self.model_data[define.AiDataKey.test_image_num] = val_class_image_num

		for k, v in history.history.items():
			if k in self.model_data:
				self.model_data[k] += v
			else:
				self.model_data[k] = v

		nlib3.save_json(Path(MODEL_DIR, self.model_name, f"{MODEL_FILE}.json"), self.model_data)
		self.show_history()
		self.show_model_test(dataset_path, max_loop_num=5)
		return self.model_data.copy()

	# モデルの学習履歴をグラフで表示する
	def show_history(self):
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

	# 学習済みのニューラルネットワークを読み込む
	def load_model(self):
		if self.model is None:
			self.model_data = nlib3.load_json(Path(MODEL_DIR, self.model_name, f"{MODEL_FILE}.json"))
			self.model = self.create_model(self.model_data[define.AiDataKey.model], self.model_data[define.AiDataKey.class_num], self.model_data[define.AiDataKey.trainable])
			self.model.load_weights(Path(MODEL_DIR, self.model_name, MODEL_FILE))
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


class ImageClassificationAi(Ai):
	def __init__(self, *args, **kwargs):
		super().__init__(define.AiType.categorical, *args, **kwargs)
		return

	# 画像分類モデルを作成する
	def create_model(self, model_type: define.ModelType, num_classes: int, trainable: bool = False):
		match model_type:
			case define.ModelType.vgg16_512:
				return self.create_model_vgg16(num_classes, trainable)
			case define.ModelType.resnet_rs152_512x2:
				return self.create_model_resnet_rs(num_classes, trainable)
		return None

	# ResNet_RSモデルを作成する
	def create_model_resnet_rs(self, num_classes, trainable):
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

		if not trainable:
			for layer in model.layers[:762]:
				layer.trainable = False

		model.compile(
			loss="categorical_crossentropy",
			optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
			metrics=["accuracy"]
		)
		return model

	# vgg16から転移学習するためのmodelを作成する
	def create_model_vgg16(self, num_classes, trainable):
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

		if not trainable:
			for layer in model.layers[:15]:
				layer.trainable = False

		# 最適化アルゴリズムをSGD（確率的勾配降下法）とし最適化の学習率と修正幅を指定してコンパイルする。
		model.compile(
			loss="categorical_crossentropy",
			optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
			metrics=["accuracy"]
		)
		return model

	# 訓練用のデータセットを生成する
	def create_dataset(self, dataset_path, batch_size, normalize = False):
		generator = self.create_generator(normalize)
		train_ds = generator.flow_from_directory(
			dataset_path,
			target_size = (self.img_height, self.img_width),
			batch_size = batch_size,
			seed=RANDOM_SEED,
			class_mode="categorical",
			subset = "training")
		val_ds = generator.flow_from_directory(
			dataset_path,
			target_size = (self.img_height, self.img_width),
			batch_size = batch_size,
			seed=RANDOM_SEED,
			class_mode="categorical",
			subset = "validation")
		return train_ds, val_ds		# [[[img*batch], [class*batch]], ...] の形式
		
	# データセットに含まれるクラスごとの画像の数を取得する
	def count_image_from_dataset(self, dataset):
		progbar = tf.keras.utils.Progbar(len(dataset))
		class_image_num = []
		for i in range(len(dataset.class_indices)):			# 各クラスの読み込み枚数を 0 で初期化して、カウント用のキーを生成する ( ３クラス中の１番目なら [1, 0, 0])
			class_image_num.append([0, [1 if i == row else 0 for row in range(len(dataset.class_indices))]])

		for i, row in enumerate(dataset):
			for image_num in class_image_num:					# 各クラスのデータ数を計測する
				image_num[0] += np.count_nonzero([np.all(x) for x in (row[1] == image_num[1])])		# numpyでキーが一致するものをカウントする
			progbar.update(i + 1)
			if i == len(dataset) - 1:							# 無限にループするため、最後まで取得したら終了する
				break
		class_image_num = [row for row, label in class_image_num]		# 不要になったラベルのキーを破棄する
		dataset.reset()
		return class_image_num, dataset.class_indices

	# 指定された画像の推論結果を取得する
	@model_required
	def inference(self, image):
		if type(image) is str:
			image = self.preprocess_image(image, self.get_normalize_flag())
		result = self.model(image)		
		return [float(row) for row in result[0]]

	# モデルから返された結果を分類のクラス名に変換する
	@model_required
	def result_to_classname(self, result):
		class_name_list = list(self.model_data[define.AiDataKey.class_indices].keys())
		return class_name_list[list(result).index(max(result))]

	# テストデータの推論結果を表示する
	@model_required
	def show_model_test(self, dataset_path, max_loop_num = 0, use_val_ds = True):
		train_ds, test_ds = self.create_dataset(dataset_path, 12, normalize=self.get_normalize_flag())
		if not use_val_ds:
			test_ds = train_ds

		for i, row in enumerate(test_ds):
			class_name_list = list(self.model_data[define.AiDataKey.class_indices].keys())
			fig = plt.figure(figsize=(16, 9))
			for j in range(len(row[0])):			# 最大12の画像数
				result = self.model(tf.expand_dims(row[0][j], 0))[0]
				ax = fig.add_subplot(3, 8, j * 2 + 1)
				ax.imshow(row[0][j])
				ax = fig.add_subplot(3, 8, j * 2 + 2)
				color = "blue"
				if list(row[1][j]).index(1) != list(result).index(max(result)):
					color = "red"
				ax.bar(np.array(range(len(class_name_list))), result, tick_label=class_name_list, color=color)
			plt.show()
			if i == len(test_ds) - 1 or i == max_loop_num - 1:
				break
		return

	# テストデータで分類に失敗したデータリストを取得する
	@model_required
	def get_model_miss_list(self, dataset_path, use_val_ds = True, print_result = False):
		train_ds, test_ds = self.create_dataset(dataset_path, 8, normalize=self.get_normalize_flag())
		result_list = []
		if not use_val_ds:
			test_ds = train_ds

		for i, row in enumerate(test_ds):
			for j in range(len(row[0])):			# 最大12の画像数
				result = self.model(tf.expand_dims(row[0][j], 0))[0]
				if list(row[1][j]).index(1) != list(result).index(max(result)):
					result_list.append(result, row[1][j])
					if print_result:
						result_class = self.result_to_classname(result)
						true_class = self.result_to_classname(row[1][j])
						print(f"{result_class} -> {true_class}")
			if i == len(test_ds) - 1:
				break
		return result_list

class ImageRegressionAi(Ai):
	def __init__(self, *args, **kwargs):
		super().__init__(define.AiType.regression, *args, **kwargs)
		return

	# 画像分類モデルを作成する
	def create_model(self, model_type: define.ModelType, num_classes: int, trainable: bool = False):
		match model_type:
			case define.ModelType.resnet_rs152_512x2_regr:
				return self.create_model_resnet_rs_regr(trainable)
		return None

	@staticmethod
	def accuracy(y_true, y_pred):
		pred_error = abs(y_true - y_pred)
		result = tf.divide(
			tf.reduce_sum(
				tf.cast(pred_error < 0.1, tf.float32)) * 0.5 + tf.reduce_sum(tf.cast(pred_error < 0.2, tf.float32)) * 0.5,		# 誤差が 0.1 以下なら 1, 0.2 以下なら 0.5 とする
				tf.cast(len(y_pred), tf.float32)
		)
		return result

	# ResNet_RSモデルを作成する
	def create_model_resnet_rs_regr(self, trainable):
		resnet = resnet_rs.ResNetRS152(include_top=False, input_shape=(self.img_height, self.img_width, 3), weights="imagenet-i224")

		top_model = tf.keras.Sequential()
		top_model.add(tf.keras.layers.Flatten(input_shape=resnet.output_shape[1:]))
		top_model.add(tf.keras.layers.Dense(512, activation="relu"))
		top_model.add(tf.keras.layers.Dense(512, activation="relu"))
		top_model.add(tf.keras.layers.Dropout(0.25))
		top_model.add(tf.keras.layers.Dense(1, activation="linear"))

		model = tf.keras.models.Model(
			inputs=resnet.input,
			outputs=top_model(resnet.output)
		)

		if not trainable:
			for layer in model.layers[:762]:
				layer.trainable = False

		model.compile(
			loss="mean_squared_error",
			optimizer=tf.keras.optimizers.Adam(learning_rate=0.00008),
			metrics=[self.accuracy]
		)
		return model

	# 訓練用のデータセットを生成する
	def create_dataset(self, data_csv_path, batch_size, normalize = False):
		generator = self.create_generator(normalize)
		df = pandas.read_csv(data_csv_path)
		df = df.sample(frac=1, random_state=0)				# ランダムに並び変える
		train_ds = generator.flow_from_dataframe(
			df,
			target_size = (self.img_height, self.img_width),
			batch_size = batch_size,
			seed=RANDOM_SEED,
			class_mode="raw",
			subset = "training")
		val_ds = generator.flow_from_dataframe(
			df,
			target_size = (self.img_height, self.img_width),
			batch_size = batch_size,
			seed=RANDOM_SEED,
			class_mode="raw",
			subset = "validation")
		return train_ds, val_ds		# [[[img*batch], [class*batch]], ...] の形式

	# データセットに含まれるクラスごとの画像の数を取得する
	def count_image_from_dataset(self, dataset):
		progbar = tf.keras.utils.Progbar(len(dataset))
		image_num = {}
			
		for i, row in enumerate(dataset):
			for value in row[1]:
				if value in image_num:
					image_num[value] += 1
				else:
					image_num[value] = 1
				
			progbar.update(i + 1)
			if i == len(dataset) - 1:							# 無限にループするため、最後まで取得したら終了する
				break
		dataset.reset()
		class_indices = {row: i for i, row in enumerate(list(image_num.keys()))}
		return list(image_num.values()), class_indices

	# 指定された画像の推論結果を取得する
	@model_required
	def inference(self, image):
		if type(image) is str:
			image = self.preprocess_image(image, self.get_normalize_flag())
		result = self.model(image)
		return float(result[0])

	# テストデータの推論結果を表示する
	@model_required
	def show_model_test(self, dataset_path, max_loop_num = 0, use_val_ds = True):
		train_ds, test_ds = self.create_dataset(dataset_path, 12, normalize=self.get_normalize_flag())
		if not use_val_ds:
			test_ds = train_ds

		for i, row in enumerate(test_ds):
			fig = plt.figure(figsize=(16, 9))
			for j in range(len(row[0])):			# 最大12の画像数
				result = self.model(tf.expand_dims(row[0][j], 0))[0]
				ax = fig.add_subplot(3, 8, j * 2 + 1)
				ax.imshow(row[0][j])
				ax = fig.add_subplot(3, 8, j * 2 + 2)
				color = "blue"
				if abs(row[1][j] - result[0]) > 0.17:
					color = "orange"
				if abs(row[1][j] - result[0]) > 0.35:
					color = "red"
				ax.bar([0, 1], [result[0], row[1][j]], tick_label=["ai", "answer"], color=color)
				ax.set_ylim(0, 1.1)
			plt.show()
			if i == len(test_ds) - 1 or i == max_loop_num - 1:
				break
		return
