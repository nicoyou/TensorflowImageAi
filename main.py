from image_classification_ai import ImageClassificationAi
from image_regression_ai import ImageRegressionAi
from define import ModelType

if __name__ == "__main__":
	dataset_path = "./dataset/"
	ic_ai = ImageClassificationAi("model_name")
	ic_ai.train_model(dataset_path, epochs=6, model_type=ModelType.vgg16_512)
	# ic_ai.load_model()
	# ic_ai.check_model_sample(anime_face)
