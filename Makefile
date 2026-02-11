train:
	git pull && uv run setup.py build_ext --inplace && uv run src/train.py 
