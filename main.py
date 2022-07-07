import ai

if __name__ == "__main__":
	anime_face = r".\image\anime_face_mini"
	age_anime_face = r".\output\dataset"
	ai.train_model(anime_face, "anime_face_mini")
	#check_model_sample(load_model("age_anime_face"), load_model_data("age_anime_face"), age_anime_face)

	#loss, acc = model.evaluate(images,  labels, verbose=2)
	#print("Restored model, accuracy: {:5.2f}%".format(100*acc))
	#model.save_weights("./model_weights.dat")
