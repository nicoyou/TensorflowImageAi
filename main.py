import tensorflow_image as tfimg

if __name__ == "__main__":
    # Image Classification
    dataset_path = "./dataset/"
    ic_ai = tfimg.ImageClassificationAi("model_name")
    ic_ai.train_model(dataset_path, epochs=6, model_type=tfimg.ModelType.vgg16_512)
    # ic_ai.load_model()
    # ic_ai.check_model_sample(anime_face)

    # pix2pix
    p2p = tfimg.PixToPix("pix2pix_model_name")
    p2p.train_model("dataset/p2p", steps=8388608)
    # p2p.load_model()
    # p2p.show_history()
