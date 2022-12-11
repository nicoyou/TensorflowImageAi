import os
from pathlib import Path

import tqdm
from PIL import Image

import tensorflow_image as tfimg

from . import AnimeFaceDetector
from . import define


class AnimeFaceAi():
    def __init__(self):
        self.afd = AnimeFaceDetector()
        return

    def crop_image_face_dir(self, in_dir: str | Path, out_dir: str | Path, overwrite: bool = False) -> int:
        """指定されたディレクトリ内のイラストの顔を切り抜いて、ディレクトリ構成を維持したまま指定されたディレクトリに出力する

        Args:
            in_dir: 読み込む画像が保存されているディレクトリ
            out_dir: 変換後の画像を保存するディレクトリ
            overwrite: 既に変換後の画像が存在した場合に、もう一度切り抜いて上書きするかどうか

        Returns:
            切り抜いた顔の数
        """
        images = tfimg.get_image_path_from_dir(in_dir)
        count = 0
        for image in tqdm.tqdm(images):
            out_path = Path(out_dir, Path(image).relative_to(in_dir))   # 画像を探したディレクトリを root としてそれ以降のパスを引き継ぐ
            if overwrite or not (out_path.parent / (out_path.stem + f"_{0:02}.png")).is_file():
                os.makedirs(out_path.parent, exist_ok=True)
                faces = self.afd.get_faces(image)

                if len(faces) >= 1:
                    pil_image = Image.open(image)
                    for i, face in enumerate(faces):
                        im_crop = pil_image.crop(face["bbox"])
                        im_crop.save(out_path.parent / (out_path.stem + f"_{i:02}.png"))
                        count += 1
        return count

    def get_face_data(self, image: str) -> dict:
        """キャラクターの顔の座標を検出して取得する

        Args:
            image: 画像のファイルパス

        Returns:
            画像情報を格納した辞書
        """
        result = {}
        faces = self.afd.get_faces(image)
        if len(faces) >= 1:
            result[define.ImageDataKey.people] = []
            for row in faces:
                face_result = {
                    define.PersonDataKey.face_score: row["score"],
                    define.PersonDataKey.face_pos: row["bbox"],
                }
                result[define.ImageDataKey.people].append(face_result)
        return result

    def get_face_data_from_imagelist(self, image_list: list[str | Path]) -> dict:
        """複数の画像のアニメ顔を検出する

        Args:
            image_list: 画像のファイルパスのリスト

        Returns:
            ファイル名をキーとして全ての画像のアニメ顔情報を格納した辞書
        """
        image_data = {}
        for row in image_list:
            try:
                result = self.get_face_data(row)
            except Exception:
                image_data[str(Path(row).name)] = {}
            else:
                image_data[str(Path(row).name)] = result
        return image_data


if __name__ == "__main__":
    import nlib3

    afai = AnimeFaceAi()
    image_dir = "./dataset"
    images = tfimg.get_image_path_from_dir(image_dir)
    nlib3.save_json(Path(image_dir, "image_data.json"), afai.get_face_data_from_imagelist(images))
    #afai.crop_image_face_dir("./dataset", "./out")
