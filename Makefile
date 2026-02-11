train:
	git pull && uv run setup.py build_ext --inplace && uv run scripts/train.py
