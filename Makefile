
download-data:
	curl https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/house_votes_84/house_votes_84.tsv.gz | gunzip - > data/house_votes_84.tsv
	curl https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/wine_recognition/wine_recognition.tsv.gz | gunzip - > data/wine_recognition.tsv
	curl https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/537_houses/537_houses.tsv.gz | gunzip - > data/537_houses.tsv

run_test: 
	python -m unittest discover -v