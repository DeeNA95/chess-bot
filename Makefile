train:
	git pull && uv run setup.py build_ext --inplace && nohup uv run scripts/train.py > training.log 2>&1 &

sync:
	rsync -avz --exclude-from='.gitignore' --exclude '.git' --exclude 'build' . vast:~/chess-bot/
