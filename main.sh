python -u main_resnet.py \
	--arch=resnet18 \
	--workers=3 \
	--epochs=40 \
	--print-freq=1 \
	--batch_size=120 |tee main.log
