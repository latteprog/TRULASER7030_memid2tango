"# TRULASER7030_memid2tango" 

Classifier for TRULASER7030 part sorter

Folder "tango" contains various scripts and text files for basic data viewing and understanding, and the implementation of neural networks.

Files:
	logs/*: results from nn-ensemble
	pca_classfier/*: saved models of nn-ensemble
	plots/*: plots used in presentation
	classifier*.py: implementations of classifier, using different features and pca preprocessing
	corr_by_feature.txt: list of correlation between each feature and success rate
	load_data.py: util script - loads and splits data into train, valid, test sets
	parts.py: view information about parts
	pca.py: preprocess and view features using pca
	success_by_name.txt: success rate for each part id
	test, train, valid.txt: list of part id for each dataset

Folder "memid" contains a Jupyter notebook, with implementations of decision trees and plots of results.

