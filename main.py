import ai

if __name__ == "__main__":
	anime_face = r"C:\Users\nicoyou\Documents\GitHub\AiCharacterImage\image\anime_face"
	age_anime_face = r"C:\Users\nicoyou\Documents\GitHub\AiCharacterImage\output\dataset"
	ai.train_model(anime_face, "anime_face")
	#check_model_sample(load_model("age_anime_face"), load_model_data("age_anime_face"), age_anime_face)


	#loss, acc = model.evaluate(images,  labels, verbose=2)
	#print("Restored model, accuracy: {:5.2f}%".format(100*acc))
	#model.save_weights("./model_weights.dat")
