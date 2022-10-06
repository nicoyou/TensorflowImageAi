import image_ai

if __name__ == "__main__":
    dataset_path = "./dataset/"
    ic_ai = image_ai.ImageClassificationAi("model_name")
    ic_ai.train_model(dataset_path, epochs=6, model_type=image_ai.ModelType.vgg16_512)
    # ic_ai.load_model()
    # ic_ai.check_model_sample(anime_face)

    # pix2pix
    p2p = image_ai.PixToPix("pix2pix_item_145")
    p2p.train_model("dataset/p2p", steps=50000000)
    # p2p.load_model()
    # p2p.show_history()
