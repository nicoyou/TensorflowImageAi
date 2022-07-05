import os
import pathlib

import tqdm
from PIL import Image

import anime_face_detector
import image_manager

def crop_image_face(in_dir: str, out_dir: str, overwrite: bool = False) -> int:
	"""指定されたディレクトリ内のイラストの顔を切り抜いて、ディレクトリ構成を維持したまま指定されたディレクトリに出力する

	Args:
		in_dir: 読み込む画像が保存されているディレクトリ
		out_dir: 変換後の画像を保存するディレクトリ
		overwrite: 既に変換後の画像が存在した場合に、もう一度切り抜いて上書きするかどうか

	Returns:
		切り抜いた顔の数
	"""
	afd = anime_face_detector.AnimeFaceDetector()
	images = image_manager.get_image_path_from_dir(in_dir)
	count = 0
	for image in tqdm.tqdm(images):
		out_path = pathlib.Path(out_dir, image)
		if overwrite or not (out_path.parent / (out_path.stem + f"_{0:02}.png")).is_file():
			os.makedirs(out_path.parent, exist_ok=True)
			faces = afd.get_faces(image)

			if len(faces) >= 1:
				pil_image = Image.open(image)
				for i, face in enumerate(faces):
					im_crop = pil_image.crop(face["bbox"])
					im_crop.save(out_path.parent / (out_path.stem + f"_{i:02}.png"))
					count += 1
	return count

if __name__ == "__main__":
	crop_image_face("./dataset", "./output")
	