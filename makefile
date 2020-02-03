documents:
	rm -f ./docs/*.html
	pdoc --html --force --output-dir docs cp_detection
	mv ./docs/cp_detection/*.html ./docs
	rmdir ./docs/cp_detection

tensorboard:
	tensorboard --logdir ./lightning_logs