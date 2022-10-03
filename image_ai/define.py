import enum
from typing import Final


class AiType(str, enum.Enum):
	categorical = "categorical"
	multi_label = "multi_label"
	regression = "regression"

	gan = "genera_tive_adversarial_networks"

class ModelType(str, enum.Enum):
	unknown = "unknown"
	vgg16_512 = "vgg16_512"
	resnet_rs152_512x2 = "resnet_rs152_512x2"
	resnet_rs152_512x2_regr = "resnet_rs152_512x2_regr"
	resnet_rs152_512x2_multi_label = "resnet_rs152_512x2_multi_label"

	pix2pix = "pix2pix"

class AiDataKey(str, enum.Enum):
	version = "version"
	model = "model"
	ai_type = "ai_type"
	trainable = "trainable"
	class_num = "class_num"
	class_indices = "class_indices"
	train_image_num = "train_image_num"
	test_image_num = "test_image_num"
	accuracy = "accuracy"
	val_accuracy = "val_accuracy"
	loss = "loss"
	val_loss = "val_loss"

class GanDataKey(str, enum.Enum):
	gen_total_loss = "gen_total_loss"
	gen_gan_loss = "gen_gan_loss"
	gen_l1_loss = "gen_l1_loss"
	disc_loss = "disc_loss"
	time = "time"


class ImageDataKey(str, enum.Enum):
	people = "people"

class PersonDataKey(str, enum.Enum):
	face_score = "face_score"
	face_pos = "face_pos"
	age = "age"


RANDOM_SEED: Final[int] = 54
