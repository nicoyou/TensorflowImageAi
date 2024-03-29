import glob
import os
from pathlib import Path

import nlib3
import tqdm
from PIL import Image


def hconcat(im1: Image, im2: Image) -> Image:
    """画像を横に結合する

    Args:
        im1: 左側のPIL画像
        im2: 右側のPIL画像

    Returns:
        結合済みのPIL画像
    """
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def vconcat(im1: Image, im2: Image) -> Image:
    """画像を縦に結合する

    Args:
        im1: 上側のPIL画像
        im2: 下側のPIL画像

    Returns:
        結合済みのPIL画像
    """
    dst = Image.new("RGB", (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def expand_to_rect_pad(pil_img: Image, aspect_ratio: tuple | list = (1, 1), background_color: tuple | list = (0, 0, 0)) -> Image:
    """余白を追加してアスペクト比を維持しながら指定されたアスペクト比の画像に変換する

    Args:
        pil_img: 変換するPIL画像
        aspect_ratio: 出力する画像のアスペクト比
        background_color: 余白の色

    Returns:
        変換後のPIL画像
    """
    aspect_ratio = nlib3.Vector2(aspect_ratio)
    img_size = nlib3.Vector2(pil_img.size)
    if img_size.x * aspect_ratio.y == img_size.y * aspect_ratio.x:
        return pil_img
    elif img_size.x * aspect_ratio.y > img_size.y * aspect_ratio.x:
        result = Image.new(pil_img.mode, (img_size.x, int(img_size.x * (aspect_ratio.y / aspect_ratio.x))), background_color)
        result.paste(pil_img, (0, int(img_size.x * (aspect_ratio.y / aspect_ratio.x) - img_size.y) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (int(img_size.y * (aspect_ratio.x / aspect_ratio.y)), img_size.y), background_color)
        result.paste(pil_img, (int(img_size.y * (aspect_ratio.x / aspect_ratio.y) - img_size.x) // 2, 0))
        return result


def expand_to_square_crop(pil_img: Image) -> Image:
    """はみ出す部分を切り取ってアスペクト比を維持しながら正方形の画像に変換する

    Args:
        pil_img: PIL画像

    Returns:
        変換後のPIL画像
    """
    crop_length = min(pil_img.size)
    data = (
        (pil_img.width - crop_length) // 2,
        (pil_img.height - crop_length) // 2,
        (pil_img.width + crop_length) // 2,
        (pil_img.height + crop_length) // 2,
    )
    return pil_img.crop(data)


def make_pix2pix_dataset(input_image_dir: str, output_image_dir: str, out_dir: str = "./") -> None:
    """ファイル名が同じ２枚の画像ペアを結合して教師データを作成する

    Args:
        input_image_dir: 入力画像が格納されたディレクトリ
        output_image_dir: 入力画像と対になる同じファイル名の出力画像が格納されたディレクトリ
        out_dir: 生成した pix2pix 用の教師データを出力するディレクトリ
    """
    i_images = glob.glob(str(Path(input_image_dir) / "**" / "*.*"), recursive=True)
    o_images = glob.glob(str(Path(output_image_dir) / "**" / "*.*"), recursive=True)

    os.makedirs(out_dir, exist_ok=True)
    old_filename = ""
    for file_path in tqdm.tqdm(i_images):
        file_name = Path(file_path).stem
        img = Image.open(file_path)
        try:
            o_img = Image.open([row for row in o_images if Path(row).stem == file_name][0])
        except Exception as e:
            print(e)
            print(file_path)
            continue

        try:
            if int(old_filename) + 1 != int(file_name):
                print(f"{file_name} は連続的ではありません")
        except Exception:
            pass

        if img.size != o_img.size:                                                                              # 画像サイズが異なれば
            img_size = nlib3.Vector2(img.size[0], img.size[1])
            o_img_size = nlib3.Vector2(o_img.size[0], o_img.size[1])
            if ((img_size / img_size.max()) * 256).floor() != ((o_img_size / o_img_size.max()) * 256).floor():  # アスペクト比も異なれば
                print(f"{file_name} の画像サイズが一致しませんでした")

        img_resized = expand_to_rect_pad(img).resize((256, 256))
        o_img_resized = expand_to_rect_pad(o_img).resize((256, 256))
        result_img = hconcat(o_img_resized, img_resized)
        result_img.save(Path(out_dir) / f"{file_name}.png")

        # img_resized = expand_to_square_crop(img).resize((256, 256))
        # o_img_resized = expand_to_square_crop(o_img).resize((256, 256))
        # result_img = hconcat(o_img_resized, img_resized)
        # result_img.save(Path(out_dir) / f"{file_name}_crop.png")

        old_filename = file_name
    return


def make_pix2pix_hd_dataset_pad(image_dir: str, out_dir: str, image_rotate: bool = True) -> None:
    """画像をpix2pixHDの教師データとして適切な画像サイズに変換する

    Args:
        image_dir: 未加工の教師データが格納されているディレクトリ
        out_dir: 変換後の教師データを保存するディレクトリ
        image_rotate: 縦長の画像を横向きに 90 度回転させるかどうか
    """
    i_images = glob.glob(str(Path(image_dir) / "**" / "*.*"), recursive=True)
    os.makedirs(out_dir, exist_ok=True)
    for file_path in tqdm.tqdm(i_images):
        file_name = Path(file_path).stem
        img = Image.open(file_path)
        if image_rotate and img.height > img.width:
            img = img.rotate(90 * 3, expand=True)

        img_resized = expand_to_rect_pad(img, (2, 1)).resize((2048, 1024))
        img_resized.save(Path(out_dir) / f"{file_name}.png")
    return


if __name__ == "__main__":
    make_pix2pix_dataset("dataset/pix2pix/input", "dataset/pix2pix/real", "dataset/out_p2p")
