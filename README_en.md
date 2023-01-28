# Tensorflow Image
### A library that makes basic image-related AI training and inference easy.
With this library, image classification, regression, multi-label classification, and pix2pix deep learning can be easily performed with a few lines of code.
Information such as the number of images used for training and the training history is also stored so that it can be viewed later or displayed in a graph. It is also possible to resume learning from where you left off.
Thus, even if learning is stopped and resumed many times, it can be displayed as one continuous learning history.

Inference results for test data can be displayed as a chart that can be compared with the parameters of the correct answer.

Each parameter that can be set for deep learning is optimized with generic values.

[![PyPI](https://img.shields.io/pypi/v/tensorflow-image)](https://pypi.org/project/tensorflow-image/)
![Python versions](https://img.shields.io/pypi/pyversions/tensorflow-image)

---

## Installation
```bash
pip install tensorflow-image
```


## USAGE ( Sample of some functions )
". /dataset/" directory to store images that will serve as teacher data.

For classification problems, images are stored in separate directories for each class, so that the directory name is recognized as the name of the classification.


For regression and multi-label classification, the path to the csv file containing the information for each image must be specified.

The structure of each csv file is as follows
filename is relative to this csv file.


### Regression analysis
|  filename       |  class  |
| --------------- | ------- |
|  ./image/a.png  |  0.3    |
|  ./image/b.png  |  0.6    |

### Multilabel classification
|  filename       |  labels                   |
| --------------- | ------------------------- |
|  ./image/a.png  |  label_a,label_c,label_d  |
|  ./image/b.png  |  label_b                  |


Training.
```python
import tensorflow_image as tfimg

# Image Classification
icai = tfimg.ImageClassificationAi("model_name_classification")
icai.train_model("./dataset/", epochs=6, model_type=tfimg.ModelType.efficient_net_v2_b0, trainable=True)

# Regression analysis
rai = tfimg.ImageRegressionAi("model_name_regression")
rai.train_model("./dataset/data.csv", epochs=6, model_type=tfimg.ModelType.efficient_net_v2_b0, trainable=True)

# Multilabel classification
mlai = tfimg.ImageMultiLabelAi("model_name_multilabel")
mlai.train_model("./dataset/data.csv", epochs=6, model_type=tfimg.ModelType.efficient_net_v2_b0, trainable=True)
```

Read and infer models.
```python
import tensorflow_image as tfimg

# Image Classification
icai = tfimg.ImageClassificationAi("model_name_classification")
icai.load_model()
result = icai.predict("./dataset/sample.png")   # infer
print(icai.result_to_classname(result))

# Regression analysis
rai = tfimg.ImageRegressionAi("model_name_regression")
rai.load_model()
result = rai.predict("./dataset/sample.png")    # infer

# Multilabel classification
mlai = tfimg.ImageMultiLabelAi("model_name_multilabel")
mlai.load_model()
result = mlai.predict("./dataset/sample.png")   # infer
print(mlai.result_to_label_dict(result))
print(mlai.result_to_labelname(result, 0.5))
```

---

## Supported Models
- MobileNetV2
- VGG16
- EfficientNetV2
- ResNet-RS


To use ResNet-RS, you must additionally pip install the following repository
> git+https://github.com/sebastian-sz/resnet-rs-keras@main

---

## Author
Nicoyou

[@NicoyouSoft](https://twitter.com/NicoyouSoft)
