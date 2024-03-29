from pathlib import Path
from typing import Final

import nlib3


class AiType(nlib3.StrEnum):
    categorical = "categorical"
    multi_label = "multi_label"
    regression = "regression"

    gan = "genera_tive_adversarial_networks"


class ModelType(nlib3.StrEnum):
    vgg16_512 = "vgg16_512"
    mobile_net_v2 = "mobile_net_v2"
    mobile_net_v2_regr = "mobile_net_v2_regr"
    mobile_net_v2_multi_label = "mobile_net_v2_multi_label"
    efficient_net_v2_b0 = "efficient_net_v2_b0"
    efficient_net_v2_s = "efficient_net_v2_s"
    efficient_net_v2_b0_regr = "efficient_net_v2_b0_regr"
    efficient_net_v2_s_regr = "efficient_net_v2_s_regr"
    efficient_net_v2_b0_multi_label = "efficient_net_v2_b0_multi_label"
    efficient_net_v2_s_multi_label = "efficient_net_v2_s_multi_label"
    resnet_rs152_256 = "resnet_rs152_256"
    resnet_rs152_512x2 = "resnet_rs152_512x2"
    resnet_rs152_256_regr = "resnet_rs152_256_regr"
    resnet_rs152_512x2_regr = "resnet_rs152_512x2_regr"
    resnet_rs152_256_multi_label = "resnet_rs152_256_multi_label"
    resnet_rs152_512x2_multi_label = "resnet_rs152_512x2_multi_label"

    pix2pix = "pix2pix"

    unknown = "unknown"


class AiDataKey(nlib3.StrEnum):
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


class GanDataKey(nlib3.StrEnum):
    gen_total_loss = "gen_total_loss"
    gen_gan_loss = "gen_gan_loss"
    gen_l1_loss = "gen_l1_loss"
    disc_loss = "disc_loss"
    time = "time"


RANDOM_SEED: Final[int] = 54
CURRENT_DIR: Final[Path] = Path().cwd()
MODEL_DIR: Final[Path] = CURRENT_DIR / "models"
MODEL_FILE: Final[str] = "model"

DEFAULT_IMAGE_SIZE: Final[int] = 224
DEFAULT_INPUT_SHAPE: Final[tuple[int, int, int]] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3)
