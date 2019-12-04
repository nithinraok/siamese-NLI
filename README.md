# snli
saved models are present in saved\_models with corresponding base or lstm

download and place data and data\_8k folder in root directory
data : https://drive.google.com/open?id=1vmyCsZolX2P4Ju3kwYY7D1uPEoZ6LXut  
data\_8k : https://drive.google.com/open?id=1FrwqHzM5Cv_sBImL3B7wwy4D-eACY8iR

to train:  
for baseline:  python train.py --epochs 40 --batch\_size 512
for lstm: python train\_lstm.py --epochs 40 --batch\_size 128

to test:
just pass --test option without epochs and batch\_size
