import tensorflow as tf

from pathlib import Path

SAVE_INTERVAL = 10
MAX_FILE = 3


# ログに学習時間を追記するコールバック
class TimeCallback(tf.keras.callbacks.Callback):
    def __init__(self, metric_name="time"):
        super().__init__()
        self.metric_name = metric_name
        self.times = []
        return

    def on_epoch_begin(self, epoch, logs=None):
        self.timetaken = tf.timestamp()
        return

    def on_epoch_end(self, epoch, logs={}):
        logs[self.metric_name] = float(tf.timestamp() - self.timetaken)
        return


# チェックポイントを保存するコールバック
class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def on_epoch_end(self, epoch, logs=None):
        if epoch % SAVE_INTERVAL == SAVE_INTERVAL - 1:
            super().on_epoch_end(epoch, logs)
            filepath = Path(self.filepath.format(epoch=epoch - SAVE_INTERVAL * MAX_FILE + 1))
            filepath.with_suffix(".data-00000-of-00001").unlink(missing_ok=True)    # MAX_FILE 数を超えたチェックポイントを削除する
            filepath.with_suffix(".index").unlink(missing_ok=True)
        else:
            self.epochs_since_last_save += 1
        return
