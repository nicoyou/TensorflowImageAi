import abc
import os
from pathlib import Path
from typing import Any, Callable, Final

CURRENT_DIR: Final[Path] = Path(__file__).parent
os.environ["PATH"] += ";" + str(CURRENT_DIR / "dll")			# 環境変数に一時的に dll のパスを追加する

import matplotlib.pyplot as plt
import nlib3
import numpy as np
import pandas
import resnet_rs
import tensorflow as tf
from PIL import ImageFile

import define
import tf_callback

__version__: Final[str] = "1.2.0"
MODEL_DIR: Final[Path] = CURRENT_DIR / "models"
MODEL_FILE: Final[str] = "model"
RANDOM_SEED: Final[int] = 54
	
def model_required(func: Callable) -> Callable:
	"""モデルが定義されている場合のみ実行するデコレーター
	"""
	def wrapper(self, *args, **kwargs):
		if self.model is None:
			nlib3.print_error_log("モデルが初期化されていません")
			return None
		return func(self, *args, **kwargs)
	return wrapper

class Ai(metaclass = abc.ABCMeta):
	"""画像分類系AIの根底クラス
	"""
	MODEL_DATA_VERSION: Final[int] = 6

	def __init__(self, ai_type: define.AiType, model_name: str) -> None:
		"""管理するAIの初期設定を行う

		Args:
			ai_type: 管理するAIの種類
			model_name: 管理するAIの名前 ( 保存、読み込み用 )
		"""
		ImageFile.LOAD_TRUNCATED_IMAGES = True			# 高速にロードできない画像も読み込む
		self.model = None
		self.model_data = None
		self.model_name = model_name
		self.ai_type = ai_type
		self.img_height = 224
		self.img_width = 224
		return

	def preprocess_image(self, img_path: Path | str, normalize: bool = False) -> tf.Tensor:
		"""画像を読み込んでテンソルに変換する

		Args:
			img_path: 読み込む画像のファイルパス
			normalize: 0 ～ 1 の範囲に正規化するかどうか

		Returns:
			画像のテンソル
		"""
		img_raw = tf.io.read_file(str(img_path))
		image = tf.image.decode_image(img_raw, channels=3)
		image = tf.image.resize(image, (self.img_height, self.img_width))
		if normalize:
			image /= 255.0						# normalize to [0,1] range
		image = tf.expand_dims(image, 0)		# 次元を一つ増やしてバッチ化する
		return image

	def pillow_image_to_tf_image(self, image: Any, normalize: bool = False) -> tf.Tensor:
		"""pillowで読み込んだ画像をテンソルに変換する

		Args:
			image: pillowイメージ
			normalize: 0 ～ 1 の範囲に正規化するかどうか

		Returns:
			画像のテンソル
		"""
		tf.keras.preprocessing.image.img_to_array(image)
		image = tf.image.resize(image, (self.img_height, self.img_width))
		if normalize:
			image /= 255.0						# normalize to [0,1] range
		image = tf.expand_dims(image, 0)		# 次元を一つ増やしてバッチ化する
		return image

	def get_normalize_flag(self, model_type: define.ModelType = define.ModelType.unknown) -> bool:
		"""画像の前処理として正規化を実行するかを判断する

		Args:
			model_type:
				正規化が必要かどうかを確認するモデル
				すでにクラスがモデルデータを保持している場合は不要

		Returns:
			正規化が必要かどうか
		"""
		normalize = True
		no_normalize_model = [define.ModelType.vgg16_512]
		if (self.model_data is not None and self.model_data[define.AiDataKey.model] in no_normalize_model) or model_type in no_normalize_model:
			normalize = False
		return normalize

	def create_generator(self, normalize: bool) -> tf.keras.preprocessing.image.ImageDataGenerator:
		"""データセットの前処理を行うジェネレーターを作成する

		Args:
			normalize: 画像の各ピクセルを0 ～ 1 の間に正規化するかどうか

		Returns:
			ジェネレーター
		"""
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

	def train_model(self, dataset_path: str, epochs: int = 6, batch_size: int = 32, model_type: define.ModelType = define.ModelType.unknown, trainable: bool = False) -> dict:
		"""ディープラーニングを実行する

		Args:
			dataset_path: 学習に使用するデータセットのファイルパス
			epochs: 学習を行うエポック数
			batch_size: バッチサイズ
			model_type:
				学習を行うモデルの種類
				モデルの新規作成時のみ指定する
				すでにモデルが読み込まれている場合はこの値は無視される
			trainable:
				転移学習を行うときに特徴検出部分を再学習するかどうか
				すでにモデルが読み込まれている場合はこの値は無視される

		Returns:
			学習を行ったモデルの情報
		"""
		train_ds, val_ds = self.create_dataset(dataset_path, batch_size, normalize=self.get_normalize_flag(model_type))

		class_image_num, class_indices = self.count_image_from_dataset(train_ds)
		val_class_image_num, val_class_indices = self.count_image_from_dataset(val_ds)

		if self.model is None:
			if model_type == define.ModelType.unknown:
				nlib3.print_error_log("モデルを新規作成する場合はモデルタイプを指定してください")
				return None
			self.model_data = {}
			self.model_data[define.AiDataKey.model] = model_type			# モデル作成時のみモデルタイプを登録する
			self.model_data[define.AiDataKey.trainable] = trainable
			self.model = self.create_model(model_type, len(class_indices), trainable)

		timetaken = tf_callback.TimeCallback()
		history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[timetaken])
		self.model.save_weights(MODEL_DIR / self.model_name / MODEL_FILE)
		self.model_data[define.AiDataKey.version] = self.MODEL_DATA_VERSION
		self.model_data[define.AiDataKey.ai_type] = self.ai_type
		self.model_data[define.AiDataKey.class_num] = len(class_indices)
		self.model_data[define.AiDataKey.class_indices] = class_indices
		self.model_data[define.AiDataKey.train_image_num] = class_image_num
		self.model_data[define.AiDataKey.test_image_num] = val_class_image_num

		for k, v in history.history.items():
			if k in self.model_data:
				self.model_data[k] += v
			else:
				self.model_data[k] = v

		nlib3.save_json(MODEL_DIR / self.model_name / f"{MODEL_FILE}.json", self.model_data)
		self.show_history()
		self.show_model_test(dataset_path, max_loop_num=5)
		return self.model_data.copy()

	def show_history(self) -> None:
		"""モデルの学習履歴をグラフで表示する
		"""
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

	def load_model(self) -> None:
		"""学習済みニューラルネットワークの重みを読み込む
		"""
		if self.model is None:
			self.model_data = nlib3.load_json(MODEL_DIR / self.model_name / f"{MODEL_FILE}.json")
			self.model = self.create_model(self.model_data[define.AiDataKey.model], self.model_data[define.AiDataKey.class_num], self.model_data[define.AiDataKey.trainable])
			self.model.load_weights(MODEL_DIR / self.model_name / MODEL_FILE)
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
	"""多クラス分類AI"""
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(define.AiType.categorical, *args, **kwargs)
		return

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
			case define.ModelType.resnet_rs152_512x2:
				return self.create_model_resnet_rs(num_classes, trainable)
		return None

	def create_model_resnet_rs(self, num_classes: int, trainable: bool) -> Any:
		"""ResNet_RSの転移学習モデルを作成する

		Args:
			num_classes: 分類するクラスの数
			trainable: 特徴量抽出部を再学習するかどうか

		Returns:
			tensorflow のモデル
		"""
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

	def create_model_vgg16(self, num_classes: int, trainable: bool) -> Any:
		"""vgg16の転移学習モデルを作成する

		Args:
			num_classes: 分類するクラスの数
			trainable: 特徴量抽出部を再学習するかどうか

		Returns:
			tensorflow のモデル
		"""
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

		# 最適化アルゴリズムをSGD ( 確率的勾配降下法 ) として最適化の学習率と修正幅を指定してコンパイルする
		model.compile(
			loss="categorical_crossentropy",
			optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
			metrics=["accuracy"]
		)
		return model

	def create_dataset(self, dataset_path: str, batch_size: int, normalize: bool = False) -> tuple:
		"""訓練用のデータセットを読み込む

		Args:
			dataset_path: 教師データが保存されているディレクトリを指定する
			batch_size: バッチサイズ
			normalize: 画像を前処理で 0 ～ 1 の範囲に正規化するかどうか

		Returns:
			train_ds: 訓練用のデータセット
			val_ds: テスト用のデータセット
		"""
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
		for i in range(len(dataset.class_indices)):					# 各クラスの読み込み枚数を 0 で初期化して、カウント用のキーを生成する ( 3 クラス中の 1 番目なら[1, 0, 0] )
			class_image_num.append([0, [1 if i == row else 0 for row in range(len(dataset.class_indices))]])

		for i, row in enumerate(dataset):
			for image_num in class_image_num:						# 各クラスのデータ数を計測する
				image_num[0] += np.count_nonzero([np.all(x) for x in (row[1] == image_num[1])])		# numpyでキーが一致するものをカウントする
			progbar.update(i + 1)
			if i == len(dataset) - 1:								# 無限にループするため、最後まで取得したら終了する
				break
		class_image_num = [row for row, label in class_image_num]	# 不要になったラベルのキーを破棄する
		dataset.reset()
		return class_image_num, dataset.class_indices

	@model_required
	def inference(self, image: str | tf.Tensor) -> list:
		"""画像を指定して推論する

		Args:
			image: 画像のファイルパスか、画像のテンソル

		Returns:
			各クラスの確立を格納したリスト
		"""
		if type(image) is str:
			image = self.preprocess_image(image, self.get_normalize_flag())
		result = self.model(image)		
		return [float(row) for row in result[0]]

	@model_required
	def result_to_classname(self, result: list | tuple) -> str:
		"""推論結果のリストをクラス名に変換する

		Args:
			result: inference 関数で得た推論結果

		Returns:
			クラス名
		"""
		class_name_list = list(self.model_data[define.AiDataKey.class_indices].keys())
		return class_name_list[list(result).index(max(result))]

	@model_required
	def show_model_test(self, dataset_path: str, max_loop_num: int = 0, use_val_ds: bool = True) -> None:
		"""テストデータの推論結果を表示する

		Args:
			dataset_path: テストに使用するデータセットのディレクトリ
			max_loop_num: 結果を表示する最大回数 ( 1 回につき複数枚の画像が表示される )
			use_val_ds: データセットから訓練用の画像を使用するかどうか ( False でテスト用データを使用する )
		"""
		train_ds, test_ds = self.create_dataset(dataset_path, 12, normalize=self.get_normalize_flag())		# バッチサイズを表示する画像数と同じにする
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
			if i == len(test_ds) - 1 or i == max_loop_num - 1:												# 表示した回数がバッチ数を超えたら終了する
				break
		return

	@model_required
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
		train_ds, test_ds = self.create_dataset(dataset_path, 8, normalize=self.get_normalize_flag())
		result_list = []
		if not use_val_ds:
			test_ds = train_ds

		for i, row in enumerate(test_ds):
			for j in range(len(row[0])):			# 最大12の画像数
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


class ImageRegressionAi(Ai):
	"""画像の回帰分析AI"""
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(define.AiType.regression, *args, **kwargs)
		return

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
			case define.ModelType.resnet_rs152_512x2_regr:
				return self.create_model_resnet_rs_regr(trainable)
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
			tf.reduce_sum(
				tf.cast(pred_error < 0.1, tf.float32)) * 0.5 + tf.reduce_sum(tf.cast(pred_error < 0.2, tf.float32)) * 0.5,		# 誤差が 0.1 以下なら 1, 0.2 以下なら 0.5 とする
				tf.cast(len(y_pred), tf.float32)
		)
		return result

	def create_model_resnet_rs_regr(self, trainable: bool) -> Any:
		"""ResNet_RSの転移学習モデルを作成する

		Args:
			trainable: 特徴量抽出部を再学習するかどうか

		Returns:
			tensorflow のモデル
		"""
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

	def create_dataset(self, data_csv_path: str, batch_size: int, normalize: bool = False) -> tuple:
		"""訓練用のデータセットを読み込む

		Args:
			data_csv_path: 教師データとなる csv ファイルのパス
			batch_size: バッチサイズ
			normalize: 画像を前処理で 0 ～ 1 の範囲に正規化するかどうか

		Returns:
			train_ds: 訓練用のデータセット
			val_ds: テスト用のデータセット
		"""
		generator = self.create_generator(normalize)
		df = pandas.read_csv(data_csv_path)
		df = df.sample(frac=1, random_state=0)				# ランダムに並び変える
		train_ds = generator.flow_from_dataframe(
			df,
			directory=Path(data_csv_path).parent,			# csv ファイルが保存されていたディレクトリを画像ファイルの親ディレクトリにする
			target_size=(self.img_height, self.img_width),
			batch_size=batch_size,
			seed=RANDOM_SEED,
			class_mode="raw",
			subset="training")
		val_ds = generator.flow_from_dataframe(
			df,
			directory=Path(data_csv_path).parent,
			target_size=(self.img_height, self.img_width),
			batch_size=batch_size,
			seed=RANDOM_SEED,
			class_mode="raw",
			subset="validation")
		return train_ds, val_ds		# [[[img*batch], [class*batch]], ...] の形式

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
			if i == len(dataset) - 1:							# 無限にループするため、最後まで取得したら終了する
				break
		dataset.reset()
		class_indices = {row: i for i, row in enumerate(list(image_num.keys()))}
		return list(image_num.values()), class_indices

	@model_required
	def inference(self, image: str | tf.Tensor) -> float:
		"""画像を指定して推論する

		Args:
			image: 画像のファイルパスか、画像のテンソル

		Returns:
			推論結果
		"""
		if type(image) is str:
			image = self.preprocess_image(image, self.get_normalize_flag())
		result = self.model(image)
		return float(result[0])

	@model_required
	def show_model_test(self, dataset_path: str, max_loop_num: int = 0, use_val_ds: bool = True) -> None:
		"""テストデータの推論結果を表示する

		Args:
			dataset_path: テストに使用するデータセットのディレクトリ
			max_loop_num: 結果を表示する最大回数 ( 1 回につき複数枚の画像が表示される )
			use_val_ds: データセットから訓練用の画像を使用するかどうか ( False でテスト用データを使用する )
		"""
		train_ds, test_ds = self.create_dataset(dataset_path, 12, normalize=self.get_normalize_flag())
		if not use_val_ds:
			test_ds = train_ds

		for i, row in enumerate(test_ds):
			fig = plt.figure(figsize=(16, 9))
			for j in range(len(row[0])):
				result = self.model(tf.expand_dims(row[0][j], 0))[0]
				ax = fig.add_subplot(3, 8, j * 2 + 1)
				ax.imshow(row[0][j])
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
