#python -u main_resnet.py \
#	--arch=resnet18 \
#	--workers=3 \
#	--epochs=100 \
#	--print-freq=1 \
#	--batch_size=140 |tee main.log

python -u main_resnet.py \
	--arch=resnet34 \
	--workers=4 \
	--print-freq=1 \
	--batch_size=100

#python -u test.py \
#	--arch=resnet18 \
#	--batch_size=1 \
#	--resumepath=../checkpoints/008_checkpoint.pth.tar \
#	| tee test.log
