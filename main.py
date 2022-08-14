import ai
import define

if __name__ == "__main__":
	anime_face = "./image/anime_face_mini"
	age_anime_face = "./output/dataset"
	ic_ai = ai.ImageClassificationAi("anime_face_mini")
	ic_ai.train_model(anime_face, epochs=6, model_type=define.ModelType.vgg16_512)
	# ic_ai.load_model()
	# ic_ai.check_model_sample(anime_face)
