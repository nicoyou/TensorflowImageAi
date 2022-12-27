import re
from codecs import open
import os

import setuptools

here = os.path.abspath(os.path.dirname(__file__))
package_name = "tensorflow-image"

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join(here, package_name.replace("-", "_"), "__init__.py")) as f:
    init_text = f.read()
    version = re.search(r'__version__:\s*Final\[str\]\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    license = re.search(r'__license__:\s*Final\[str\]\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    author = re.search(r'__author__:\s*Final\[str\]\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    author_email = re.search(r'__author_email__:\s*Final\[str\]\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    url = re.search(r'__url__:\s*Final\[str\]\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)

assert version
assert license
assert author
assert author_email
assert url

setuptools.setup(
    name=package_name,                                          # パッケージ名 ( プロジェクト名 )
    packages=[package_name.replace("-", "_")],                  # パッケージ内 ( プロジェクト内 ) のパッケージ名をリスト形式で指定 ( ここを指定しないとパッケージが含まれずに、テキストのみのパッケージになってしまう )
    version=version,                                            # バージョン
    license=license,                                            # ライセンス
    install_requires=[                                          # pip installする際に同時にインストールされるパッケージ名をリスト形式で指定
        "nlib3",
        "tensorflow",
        "numpy",
        "tqdm",
        "opencv-contrib-python",
        "pillow",
        "matplotlib",
        "pandas",
        "scipy",
    ],
    author=author,                                              # パッケージ作者の名前
    author_email=author_email,                                  # パッケージ作者の連絡先メールアドレス
    url=url,                                                    # パッケージに関連するサイトのURL ( GitHubなど )
    description="Can train and infer image-related AI",         # パッケージの簡単な説明
    long_description=long_description,                          # PyPIに"Project description"として表示されるパッケージの説明文
    long_description_content_type="text/markdown",              # long_descriptionの形式を"text/plain", "text/x-rst", "text/markdown"のいずれかから指定
    keywords=f"{package_name} tensorflow image ai nicoyou",     # PyPIでの検索用キーワードをスペース区切りで指定
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
    ],                                                          # パッケージ ( プロジェクト ) の分類 ( https://pypi.org/classifiers/ )
)
