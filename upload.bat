call .venv/Scripts/activate.bat

@rem https://test.pypi.org/manage/projects/
twine upload --repository testpypi dist/*

@rem https://pypi.org/manage/projects/
twine upload --repository pypi dist/*

pause
