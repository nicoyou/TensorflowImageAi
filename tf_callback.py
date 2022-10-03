import tensorflow as tf


class TimeCallback(tf.keras.callbacks.Callback):
	def __init__(self, metric_name = "time"):
		self.metric_name = metric_name
		self.times = []
		return
	def on_epoch_begin(self, epoch, logs=None):
		self.timetaken = tf.timestamp()
		return
	def on_epoch_end(self, epoch, logs = {}):
		logs[self.metric_name] = float(tf.timestamp() - self.timetaken)
		return
