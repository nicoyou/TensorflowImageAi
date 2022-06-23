import base64				# base64の変換を行う
import datetime				# 日付
import enum					# 列挙子
import inspect				# 活動中のオブジェクトの情報を取得する ( エラー位置 )
import json					# JSONファイルを扱う
import os					# osの情報
import platform				# OS情報
import re					# 正規表現
import subprocess			# 外部プログラム実行用
import sys					# Pythonバージョン
import threading			# マルチスレッド
import time					# sleepなどの時間系
import traceback			# スタックトレースの取得
import urllib.error			# urllibのエラー定義
import urllib.request		# urlを扱うモジュール
from typing import Any, Union, Callable

__version__ = "1.13.0+"
OUTPUT_DIR = "./data"								# 情報を出力する際のディレクトリ
LOG_PATH = OUTPUT_DIR + "/lib.log"					# ログのファイルパス
ERROR_LOG_PATH = OUTPUT_DIR + "/error.log"			# エラーログのファイルパス
DISPLAY_DEBUG_LOG_FLAG = True						# デバッグログを出力するかどうか
DEFAULT_ENCODING = "utf-8"							# ファイルIO時の標準エンコード

JsonValue = Union[int, float, bool, str]

class LibErrorCode(enum.Enum):
	"""ライブラリ内の一部関数で返されるエラーコード
	"""
	success = enum.auto()			# 成功
	file_not_found = enum.auto()	# ファイルが見つからなかった
	http = enum.auto()				# http通信のエラー
	argument = enum.auto()			# 引数が原因のエラー
	cancel = enum.auto()			# 前提条件不一致で処理がキャンセルされたときのエラー
	unknown = enum.auto()			# 不明なエラー
	
class Vector2():
	""" ２次元ベクトルの値を格納するためのクラス
	Vector2.x と Vector2.y か Vector2[0] と Vector2[1] でそれぞれの値にアクセスできる
	"""
	def __init__(self, x: Union[int, float] = 0, y: Union[int, float] = 0) -> None:
		"""それぞれの値を初期化する、値を指定しなかった場合は０で初期化される

		Args:
			x: 数値を指定する
			y: 数値を指定する
		"""
		self.x = x
		self.y = y
		return

	def __str__(self):
		return "x={}, y={}".format(self.x, self.y)
	def __repr__(self):
		return self.__str__()
	def __bool__(self):
		return (bool(self.x) or bool(self.y))
	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return (self.x == other.x and self.y == other.y)
		else:
			kls = other.__class__.__name__
			raise NotImplementedError(f"comparison between Vector2 and {kls} is not supported")
	def __ne__(self, other):
		return not self.__eq__(other)
	def __len__(self):
		return 2
	def __getitem__(self, index):
		if index == 0 or index == "x":
			return self.x
		elif index == 1 or index == "y":
			return self.y
		raise IndexError

class JsonData():
	""" Jsonファイルから一つの値を読み込んで保持するクラス
	"""
	def __init__(self, keys: Union[str, list, tuple], default: JsonValue, path: str) -> None:
		"""Jsonファイルから読み込む値を指定する

		Args:
			keys: Jsonデータのキーを指定する (複数階層ある場合はリストで渡す)
			default: 値が存在しなかった場合のデフォルトの値を設定する
			path: Jsonファイルのパス
		"""
		self.keys = keys
		self.default = default
		self.path = path
		self.data = None
		self.load_error_flag = False
		self.load()
		return

	def load(self) -> bool:
		"""ファイルから値を読み込む

		Returns:
			bool:
				正常に読み込めた場合か、デフォルト値で初期化した場合は True
				何らかのエラーが発生した場合は False
		"""
		try:
			json_data = load_json(self.path)
			if type(self.keys) is not list and type(self.keys) is not tuple:
				self.keys = (self.keys,)							# タプルでもリストでもなければタプルに加工する
			try:
				for row in self.keys:
					json_data = json_data[row]						# キーの名前をたどっていく
				self.data = json_data
				return True
			except KeyError as e:
				self.data = self.default							# キーが見つからなければデフォルト値を設定する
				print_debug(e)
				return True
		except FileNotFoundError as e:								# ファイルが見つからなかった場合はデフォルト値を設定する
			self.data = self.default
			return True
		except Exception as e:
			self.data = self.default
			self.load_error_flag = True
			print_error_log("jsonファイルの読み込みに失敗しました [keys={}]\n{}".format(self.keys, e))
		return False

	def save(self) -> bool:
		"""ファイルに現在保持している値を保存する

		Returns:
			bool: ファイルへの保存が成功した場合は True
		"""
		if self.load_error_flag:
			print_error_log("データの読み込みに失敗しているため、上書き保存をスキップしました")
			return False
		json_data = {}
		try:
			json_data = load_json(self.path)
		except FileNotFoundError as e:						# ファイルが見つからなかった場合は
			print_log("jsonファイルが見つからなかったため、新規生成します [keys={}]\n{}".format(self.keys, e))
		except json.decoder.JSONDecodeError as e:			# JSONの文法エラーがあった場合は新たに上書き保存する
			print_log("jsonファイルが壊れている為、再生成します [keys={}]\n{}".format(self.keys, e))
		except Exception as e:								# 不明なエラーが起きた場合は上書きせず終了する
			print_error_log("jsonファイルへのデータの保存に失敗しました [keys={}]\n{}".format(self.keys, e))
			return False
		try:
			update_nest_dict(json_data, self.keys, self.data)
			save_json(self.path, json_data)
			return True
		except Exception as e:
			print_error_log("jsonへの出力に失敗しました [keys={}]\n{}".format(self.keys, e))
		return False

	def increment(self, save_flag: bool = False, num: int = 1) -> bool:
		"""値をインクリメントしてファイルに保存する ( 数値以外が保存されていた場合は 0 で初期化 )

		Args:
			save_flag: ファイルにデータを保存するかどうかを指定する
			num: 増加させる値を指定する

		Returns:
			bool: データがファイルに保存されれば True
		"""
		if not can_cast(self.get(), int):							# int型に変換できない場合は初期化する
			self.set(0)
		return self.set(int(self.get()) + num, save_flag)			# 一つインクリメントして値を保存する

	def get(self) -> JsonValue:
		"""現在保持している値を取得する

		Returns:
			JsonValue: 保持している値
		"""
		return self.data

	def set(self, data: JsonValue, save_flag: bool = False) -> bool:
		"""新しい値を登録する

		Args:
			data: 新しく置き換える値
			save_flag: ファイルに新しい値を保存するかどうか

		Returns:
			bool: データがファイルに保存されれば True
		"""
		self.data = data
		if save_flag:
			return self.save()				# 保存フラグが立っていれば保存する
		return False						# 保存無し

	def get_keys(self) -> tuple:
		"""Jsonファイルのこの値が保存されているキーを取得する

		Returns:
			tuple: 値にたどり着くまでのキー
		"""
		return tuple(self.keys)

	def get_default(self) -> JsonValue:
		"""設定されているデフォルト値を取得する

		Returns:
			JsonValue: ファイルに値が存在しなかった時に使用するデフォルト値
		"""
		return self.default

	@staticmethod
	def dumps(json_data: Union[str, dict]) -> str:
		"""Jsonファイルか辞書を整形されたJson形式の文字列に変換する

		Args:
			json_data: Jsonファイルのファイルパスか、出力したいデータの辞書

		Returns:
			str: 整形されたJson形式の文字列
		"""
		if type(json_data) is str:
			data = json.loads(json_data)
		elif type(json_data) is dict:
			data = json_data
		else:
			print_error_log("JSONデータの読み込みに失敗しました")
			return None

		data_str = json.dumps(data, indent=4, ensure_ascii=False)
		return data_str


def thread(func: Callable) -> Callable:
	"""関数をマルチスレッドで実行するためのデコレーター
	"""
	def inner(*args, **kwargs) -> threading.Thread:
		th = threading.Thread(target=lambda: func(*args, **kwargs))
		th.start()
		return th
	return inner


def make_lib_dir() -> None:
	"""ライブラリ内で使用するディレクトリを作成する
	"""
	os.makedirs(OUTPUT_DIR, exist_ok=True)		# データを出力するディレクトリを生成する
	return

def get_error_message(code: LibErrorCode) -> str:
	"""ライブラリ内エラーコードからエラーメッセージを取得する

	Args:
		code: ライブラリのエラーコード

	Returns:
		str: コードに対応するエラーメッセージ
	"""
	if code == LibErrorCode.success:
		return "処理が正常に終了しました"
	elif code == LibErrorCode.file_not_found:
		return "ファイルが見つかりませんでした"
	elif code == LibErrorCode.http:
		return "HTTP通信関係のエラーが発生しました"
	elif code == LibErrorCode.argument:
		return "引数が適切ではありません"
	elif code == LibErrorCode.cancel:
		return "処理がキャンセルされました"
	elif code == LibErrorCode.unknown:
		return "不明なエラーが発生しました"
	else:
		print_error_log("登録されていないエラーコードが呼ばれました", console_print=False)
	return "不明なエラーが発生しました"

def print_log(message: object, console_print: bool = True, error_flag: bool = False, file_name: str = "", file_path: str = "") -> bool:
	"""ログをファイルに出力する

	Args:
		message: ログに出力する内容
		console_print: コンソール出力するかどうか
		error_flag: 通常ログではなく、エラーログに出力するかどうか
		file_name: 出力するファイル名を指定する ( 拡張子は不要 )
		file_path: 出力するファイルのパスを指定する

	Returns:
		bool: 正常にファイルに出力できた場合は True
	"""
	log_path = LOG_PATH
	if error_flag:					# エラーログの場合はファイルを変更する
		log_path = ERROR_LOG_PATH
	if file_name:
		log_path = os.path.join(OUTPUT_DIR, f"{file_name}.log")
	if file_path:
		log_path = file_path
	if console_print:
		print_debug(message)
	if file_name and file_path:
		raise ValueError
	
	time_now = get_datatime_now(True)					# 現在時刻を取得する
	if not os.path.isfile(log_path) or os.path.getsize(log_path) < 1024*1000*50:		# 50MBより小さければ出力する
		os.makedirs(OUTPUT_DIR, exist_ok=True)											# データを出力するディレクトリを生成する
		with open(log_path, mode="a", encoding=DEFAULT_ENCODING) as f:
			if error_flag:		# エラーログ
				frame = inspect.currentframe().f_back.f_back				# 関数が呼ばれた場所の情報を取得する
				try:
					class_name = str(frame.f_locals["self"])
					class_name = re.match(r'.*?__main__.(.*?) .*?', class_name)
					if class_name is not None:
						class_name = class_name.group(1)
				except KeyError:											# クラス名が見つからなければ
					class_name = None
				err_file_name = os.path.splitext(os.path.basename(frame.f_code.co_filename))[0]

				code_name = ""
				if class_name is not None:
					code_name = "{}.{}.{}({})".format(err_file_name, class_name, frame.f_code.co_name, frame.f_lineno)
				else:
					code_name = "{}.{}({})".format(err_file_name, frame.f_code.co_name, frame.f_lineno)
				f.write("[{}] {}".format(time_now, code_name).ljust(90)
				+ str(message).rstrip("\n").replace("\n", "\n" + "[{}]".format(time_now).ljust(90)) + "\n")		# 最後の改行文字を取り除いて文中の改行前にスペースを追加する
			else:						# 普通のログ
				f.write("[{}] {}\n".format(time_now, str(message).rstrip("\n")))
			return True
	else:
		print_debug("ログファイルの容量がいっぱいの為、出力を中止しました")
	return False

def print_error_log(message: object, console_print: bool = True) -> bool:
	"""エラーログを出力する

	Args:
		message: ログに出力する内容
		console_print: 内容をコンソールに出力するかどうか

	Returns:
		bool: 正常にファイルに出力できた場合は True
	"""
	return print_log(message, console_print, True)

def print_debug(message: object, end: str = "\n") -> bool:
	"""デバッグログをコンソールに出力する

	Args:
		message: 出力する内容
		end: 最後に追加で出力される内容

	Returns:
		bool: 実際にコンソールに出力された場合は True
	"""
	if DISPLAY_DEBUG_LOG_FLAG:
		print(message, end=end)
	return DISPLAY_DEBUG_LOG_FLAG

def load_json(file_path: str) -> dict:
	"""jsonファイルを読み込む

	Args:
		file_path: jsonファイルパス

	Returns:
		dict: 読み込んだjsonファイルのデータ
	"""
	with open(file_path, "r", encoding=DEFAULT_ENCODING) as f:
		data = json.load(f)
	return data

def save_json(file_path: str, data: dict, ensure_ascii: bool = False) -> None:
	"""辞書データをjsonファイルに保存する

	Args:
		file_path: jsonファイルパス
		data: 保存するデータ
		ensure_ascii: 非ASCII文字文字をエスケープする
	"""
	with open(file_path, "w", encoding=DEFAULT_ENCODING) as f:
		json.dump(data, f, indent=4, ensure_ascii=ensure_ascii)
	return

def update_nest_dict(dictionary: dict, keys: Union[object, list, tuple], value: object) -> bool:
	"""ネストされた辞書内の特定の値のみを再帰で変更する関数

	Args:
		dictionary: 更新する辞書
		keys: 更新する値にたどり着くまでのキーを指定し、複数あればlistかtupleで指定する
		value: 上書きする値

	Returns:
		bool: 再帰せずに更新した場合のみ True を返し、再帰した場合は False を返す
	"""
	if type(keys) is not list and type(keys) is not tuple:
		keys = (keys,)													# 渡されがキーがリストでもタプルでもなければタプルに変換する
	if len(keys) == 1:
		dictionary[keys[0]] = value										# 最深部に到達したら値を更新する
		return True
	if keys[0] in dictionary:
		update_nest_dict(dictionary[keys[0]], keys[1:], value)			# すでにキーがあればその内部から更に探す
	else:
		dictionary[keys[0]] = {}										# キーが存在しなければ空の辞書を追加する
		update_nest_dict(dictionary[keys[0]], keys[1:], value)
	return False

def check_url(url: str) -> bool:
	"""リンク先が存在するかどうかを確認する

	Args:
		url: 存在を確認するURL

	Returns:
		bool: リンク先に正常にアクセスできた場合は True
	"""
	try:
		headers = {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
		}
		req = urllib.request.Request(url, None, headers)
		f = urllib.request.urlopen(req)
		f.close()
		time.sleep(0.1)
	except Exception:
		return False		# 失敗
	return True				# 成功

def download_file(url: str, dest_path: str, overwrite: bool = True) -> LibErrorCode:
	"""インターネット上からファイルをダウンロードする

	Args:
		url: ダウンロードするファイルのURL
		dest_path: ダウンロードしたファイルを保存するローカルファイルパス
		overwrite: 同名のファイルが存在した場合に上書きするかどうか

	Returns:
		LibErrorCode: ライブラリのエラーコード
	"""
	if not overwrite and os.path.isfile(dest_path):
		return LibErrorCode.cancel

	try:
		headers = {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
		}
		req = urllib.request.Request(url, None, headers)
		with urllib.request.urlopen(req) as web_file:
			data = web_file.read()
			with open(dest_path, mode="wb") as local_file:
				local_file.write(data)
				time.sleep(0.1)
				return LibErrorCode.success
	except urllib.error.HTTPError as e:
		print_error_log(f"{e} [url={url}]")
		return LibErrorCode.argument		# HTTPエラーが発生した場合は引数エラーを返す
	except (urllib.error.URLError, TimeoutError) as e:
		print_error_log(f"{e} [url={url}]")
		return LibErrorCode.http
	except FileNotFoundError as e:
		print_error_log(f"{e} [url={url}]")
		return LibErrorCode.file_not_found
	return LibErrorCode.unknown

def download_and_check_file(url: str, dest_path: str, overwrite: bool = True, trial_num: int = 3, trial_interval: int = 3) -> LibErrorCode:
	"""ファイルをダウンロードして、失敗時に再ダウンロードを試みる

	Args:
		url: ダウンロードするファイルのURL
		dest_path: ダウンロードしたファイルを保存するローカルファイルパス
		overwrite: 同名のファイルが存在した場合に上書きするかどうか
		trial_num: 最初の一回を含むダウンロード失敗時の再試行回数
		trial_interval: ダウンロード再試行までのクールタイム

	Returns:
		LibErrorCode: ライブラリのエラーコード
	"""
	result = download_file(url, dest_path, overwrite)
	if result in [LibErrorCode.cancel, LibErrorCode.argument, LibErrorCode.file_not_found]:		# 既にファイルが存在した場合と引数が間違えている場合は処理を終了する
		return result
	for i in range(trial_num):
		if not os.path.isfile(dest_path):
			print_debug(f"ダウンロードに失敗しました、{trial_interval}秒後に再ダウンロードします ( {i + 1} Fail )")
			time.sleep(trial_interval)
			result = download_file(url, dest_path, overwrite)	# 一度目はエラーコードに関わらず失敗すればもう一度ダウンロードする
			if result == LibErrorCode.argument:					# URLが間違っていれば処理を終了する
				return result
		elif result == LibErrorCode.success:					# ダウンロード成功
			return LibErrorCode.success
	return LibErrorCode.unknown

def read_tail(path: str, n: int, encoding: bool = None) -> str:
	"""ファイルを後ろから指定した行だけ読み込む

	Args:
		path: 読み込むファイルのファイルパス
		n: 読み込む行数
		encoding: ファイルのエンコード

	Returns:
		str: 実際に読み込んだ結果
	"""
	try:
		with open(path, "r", encoding=encoding) as f:
			lines = f.readlines()			# すべての行を取得する
	except FileNotFoundError:
		lines = []
	return lines[-n:]						# 後ろからn行だけ返す

def rename_path(file_path: str, dest_name: str, up_hierarchy_num: int = 0, slash_only: bool = False) -> str:
	"""ファイルパスの指定した階層をリネームする

	Args:
		file_path: リネームするファイルパス
		dest_name: 変更後のディレクトリ名
		up_hierarchy_num: 変更するディレクトリの深さ ( 一番深いディレクトリが０ )
		slash_only: パスの区切り文字をスラッシュのみにするかどうか

	Returns:
		str: 変換後のファイルパスを返す
	"""	
	file_name = ""
	for i in range(up_hierarchy_num):				# 指定された階層分だけパスの右側を避難する
		if i == 0:
			file_name = os.path.basename(file_path)
		else:
			file_name = os.path.join(os.path.basename(file_path), file_name)
		file_path = os.path.dirname(file_path)

	file_path = os.path.dirname(file_path)			# 一番深い階層を削除する
	file_path = os.path.join(file_path, dest_name)					# 一番深い階層を新しい名前で追加する
	if file_name != "":
		file_path = os.path.join(file_path, file_name)				# 避難したファイルパスを追加する
	if slash_only:
		file_path = file_path.replace("\\", "/")					# 引数で指定されていれば区切り文字をスラッシュで統一する
	return file_path

# JANコードのチェックデジットを計算して取得する
def get_check_digit(jan_code: Union[int, str]) -> int:
	"""JANコードのチェックデジットを計算して取得する

	Args:
		jan_code: 13桁のJANコードか、その最初の12桁

	Returns:
		int: 13桁目のチェックデジット
	"""
	if not type(jan_code) is str:
		jan_code = str(jan_code)
	if len(jan_code) == 13:
		jan_code = jan_code[:12]
	if len(jan_code) != 12:
		return None

	try:
		even_sum = 0
		odd_sum = 0
		for i in range(12):
			if (i + 1) % 2 == 0:
				even_sum += int(jan_code[i])						# 偶数桁の合計
			else:
				odd_sum += int(jan_code[i])							# 奇数桁の合計
		check_digit = (10 - (even_sum * 3 + odd_sum) % 10) % 10		# チェックデジット
	except Exception as e:
		print_error_log(e)
		return None
	return check_digit

def program_pause(program_end: bool = True) -> None:
	"""入力待機でプログラムを一時停止する関数

	Args:
		program_end: 再開した時にプログラムを終了する場合は True、処理を続ける場合は False
	"""
	if not False:	#__debug__:			# デバッグでなければ一時停止する
		if program_end:
			message = "Press Enter key to exit . . ."
		else:
			message = "Press Enter key to continue . . ."
		input(message)
	return

def imput_while(str_info: str, branch: Callable[[str], bool] = lambda in_str : in_str != "") -> str:
	"""条件に一致する文字が入力されるまで再入力を求める入力関数 ( デフォルトでは空白のみキャンセル )

	Args:
		str_info: 入力を求める時に表示する文字列
		branch: 正常な入力かどうかを判断する関数

	Returns:
		str: 入力された文字列
	"""
	while True:
		in_str = input(str_info)
		if branch(in_str):
			return in_str
		else:
			print("\n不正な値が入力されました、再度入力して下さい")
	return ""

def get_datatime_now(to_str: bool = False) -> Union[datetime.datetime, str]:
	"""日本の現在の datetime を取得する

	Args:
		to_str: 文字列に変換して取得するフラグ

	Returns:
		datetime | str: 日本の現在時間を datetime 型か文字列で返す
	"""
	datetime_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), "JST"))		# 日本の現在時刻を取得する
	if not to_str:
		return datetime_now
	return datetime_now.strftime("%Y-%m-%d %H:%M:%S")												# 文字列に変換する

def compress_hex(hex_str: str, decompression: bool = False) -> str:
	"""16進数の文字列を圧縮、展開する

	Args:
		hex_str: 16進数の値
		decompression: 渡された値を圧縮ではなく展開するフラグ

	Returns:
		str: 圧縮 or 展開した文字列
	"""
	if decompression:														# 展開が指定されていれば展開する
		if type(hex_str) is not str:
			return ""														# 文字列以外が渡されたら空白の文字列を返す
		hex_str = hex_str.replace("-", "+").replace("_", "/")				# 安全な文字列をbase64の記号に復元する
		hex_str += "=" * (len(hex_str) % 4)									# 取り除いたパディングを復元する
		hex_str = hex_str.encode()

		hex_str = base64.b64decode(hex_str)
		hex_str = base64.b16encode(hex_str)
		return hex_str.decode()

	if type(hex_str) is str:
		hex_str = hex_str.encode()			# バイナリデータでなければバイナリに変換する
	if len(hex_str) % 2 != 0:
		hex_str = b"0" + hex_str			# 奇数の場合は先頭に0を追加して偶数にする

	hex_str = base64.b16decode(hex_str, casefold=True)
	hex_str = base64.b64encode(hex_str)
	return hex_str.decode().replace("=", "").replace("+", "-").replace("/", "_")			# パディングを取り除いて安全な文字列に変換する

def subprocess_command(command: str) -> str:
	"""OSのコマンドを実行する

	Args:
		command: 実行するコマンド

	Returns:
		str: 実行結果
	"""
	if platform.system() == "Windows":									# Windowsの環境ではコマンドプロンプトを表示しないようにする
		si = subprocess.STARTUPINFO()
		si.dwFlags |= subprocess.STARTF_USESHOWWINDOW					# コマンドプロンプトを表示しない
		return subprocess.check_output(command, startupinfo=si)
	else:																# STARTUPINFO が存在しない OS があるため処理を分岐する
		return subprocess.check_output(command)

def print_exc() -> None:
	"""スタックされているエラーを表示する
	"""
	if DISPLAY_DEBUG_LOG_FLAG:
		traceback.print_exc()
		print_debug("\n")
		print_debug(sys.exc_info())
	return

def can_cast(x: Any, cast_type: Callable) -> bool:
	"""指定された値がキャストできるかどうかを確認する

	Args:
		x: 確認する値
		cast_type: チェックするためのキャスト関数

	Returns:
		bool: キャストできる場合は True
	"""
	try:
		cast_type(x)
	except ValueError:
		return False
	return True

def get_python_version() -> str:
	"""Pythonのバージョン情報を文字列で取得する

	Returns:
		str: Pythonのバージョン
	"""
	version = "{}.{}.{}".format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
	return version


'''
----------------------------------------------------------------------------------------------------------
ver.1.14.0 (2022//)
load_json() 関数と save_json() 関数を追加
JsonData() クラス内のファイルIO処理を全て新規追加した関数に変更
ファイルIO時の標準エンコードを定数に追加

----------------------------------------------------------------------------------------------------------
ver.1.13.0 (2022/07/05)
get_check_digit() 関数を追加
サードパーティー製ライブラリを使用していた convert_file_encoding() 関数を削除
download_file() 関数と、download_and_check_file() 関数で対応できるエラーを追加
download_and_check_file() 関数に再試行に関するパラメーターを指定できる引数を追加、返り値の型を変更
エラーコード名 LibErrorCode.file を LibErrorCode.file_not_found に変更

----------------------------------------------------------------------------------------------------------
ver.1.12.0 (2022/06/07)
全クラスと関数に引数の型と返り値の型を追加、トップのコメントをdocstringに書き換え
Vector2クラスの仕様を変更し、コンストラクタで x しか値が与えられなかった場合は y が 0 で初期化されるように変更
print_log()関数にログを出力するファイル名を指定する事ができる引数を追加
マルチスレッドのデコレータ関数がスレットを返すように変更

関数内のデバッグ出力で print_debug() 関数を使用せずに print() 関数を使用していた箇所がいくつかあった不具合を修正
ファイルをダウンロードする関数が 302 エラーを稀に起こす対策としてユーザーエージェントを追加

----------------------------------------------------------------------------------------------------------
ver.1.11.0 (2021/10/03)
ファイルパスの指定した階層をリネームする関数を追加
OSのコマンドを実行する関数を追加
print_debug 関数に引数 end を追加

----------------------------------------------------------------------------------------------------------
ver.1.10.1 (2021/07/16)
JsonData()クラスで、同時に同じファイルを更新しようとすると、ファイルのデータが初期化される不具合を修正
返り値の値を調整、JsonData.incrementをリファクタリング

----------------------------------------------------------------------------------------------------------
ver.1.10.0 (2021/06/06)
JSON文字列を整形して print 出力する関数を JsonData クラスに追加
関数をマルチスレッドで実行するためのデコレーターを追加
print_log関数でクラス名が存在しないときにクラッシュすることがある不具合を修正
PyLintを使用して、プログラムをリファクタリング

----------------------------------------------------------------------------------------------------------
ver.1.9.0 (2021/04/24)
２次元ベクトルの値を保存する構造体を追加
read_tail 関数で encoding を指定できるように引数を追加
print_log 関数のエラーログを出力する際の引数を bool 値のフラグを指定するだけで出力できるように簡略化

----------------------------------------------------------------------------------------------------------
ver.1.8.0 (2021/04/09)
エラーログに出力されるソースコードの情報にファイル名とクラス名を追加
エラーログが複数行の場合は、２行目以降の左の空白に日時が出力されるように変更
ログファイルの最大容量を 10MB から 50MB まで増加

----------------------------------------------------------------------------------------------------------
ver.1.7.0 (2021/03/13)
スタックされているエラーを表示する関数を追加
指定されたファイルが指定された文字コードでなければ指定された文字コードに変換する関数を追加
ネストされた辞書内の特定の値のみを再帰で変更する関数を JsonData クラス内からライブラリの関数に移動
ログを出力する際に OS のデフォルト文字コードで出力していたのを utf-8 で出力するように変更
JsonData クラスに値をインクリメントしてファイルに保存する関数を追加
JsonData クラスで扱うファイルの文字コードを OS のデフォルト文字コードから utf-8 に変更 ( 過去verで保存したファイルの読み込み不可 )
JsonData クラスで読み込んだJsonファイルに文法エラーがあった場合に、データを新しく上書き保存できない不具合を修正

----------------------------------------------------------------------------------------------------------
ver.1.6.0 (2021/03/07)
Jsonデータを読み込んで保持するクラスで、自由な数のキーを指定できる用に変更
キャストできるかどうかを確認する関数を追加

----------------------------------------------------------------------------------------------------------
ver.1.5.0 (2021/02/15)
ファイルを後ろから指定した行だけ読み込む関数を追加
16進数の文字列を圧縮、展開する関数を追加

----------------------------------------------------------------------------------------------------------
ver.1.4.0 (2021/01/12)
Pythonのバージョン情報を文字列で取得する関数を追加
デバッグログを出力する関数を追加
ログに出力する日付の計算に get_datatime_now 関数を使うように変更

----------------------------------------------------------------------------------------------------------
ver.1.3.1 (2020/12/24)
get_datatime 関数の名前を get_datatime_now に変更、文字列に変換した際のフォーマットを変更

----------------------------------------------------------------------------------------------------------
ver.1.3.0 (2020/12/01)
日本の現在の datetime を取得する関数を追加
Jsonデータを読み込んで保持するクラスを追加
初期化関数をライブラリ内で使用するディレクトリを作成する関数に名前を変更

----------------------------------------------------------------------------------------------------------
ver.1.2.1 (2020/11/24)
ログを出力する際に、元のメッセージの最後に改行が含まれていた場合は改行を削除するように調整
エラーログを出力する関数を複数行のログに対応
起動時の初期化処理を行う関数を作成

----------------------------------------------------------------------------------------------------------
ver.1.2.0 (2020/10/29)
条件に一致する文字が入力されるまで再入力を求める入力関数を追加

----------------------------------------------------------------------------------------------------------
ver.1.1.1 (2020/10/20)
download_file関数とdownload_and_check_file関数でファイルが既に存在する場合に上書きするかどうかを指定できる引数を追加

----------------------------------------------------------------------------------------------------------
ver.1.1.0 (2020/10/17)
通常のログを出力する関数を追加
プログラム終了時に一時停止する関数を追加
リンク先が存在するかどうかを確認する関数を追加
ライブラリ内エラーコードからエラーメッセージを取得する関数を追加
列挙型のライブラリ内エラーコードを追加

インターネット上からファイルをダウンロードする関数で、存在しないURLを指定するとクラッシュする不具合を修正
print_log関数に文字列以外のものを渡した際にクラッシュする不具合を修正

----------------------------------------------------------------------------------------------------------
ver.1.0.0 (2020/10/15)
初版
エラーログを出力する関数を実装
インターネット上からファイルをダウンロードする関数を追加
ファイルをダウンロードして、失敗時に再ダウンロードを試みる関数
'''
