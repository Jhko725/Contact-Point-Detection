documents:
	rm -f ./docs/*.html
	pdoc --html --force --output-dir docs cp_detection
	mv ./docs/cp_detection/*.html ./docs
	rmdir ./docs/cp_detection

tensorboard:
	tensorboard --logdir ./lightning_logs

clear_logs:
	rm -f ./lightning_logs/*

clear_ckpt:
	rm -f ./checkpoints/*