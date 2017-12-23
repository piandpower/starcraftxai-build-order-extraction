import pandas as pd
import numpy as np   
import json
from itertools import izip
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import multiprocessing

def load_all_build_orders():
	build_orders = []
	with open("build_orders.json",'r') as infile:
		for line in infile:
			build_data = json.loads(line)
			build_orders.append(build_data)
			races = [build_data[2],build_data[3]]
	return build_orders

def build_dataset():
	# load build orders
	build_orders = load_all_build_orders()

	# load all friendly builds and their label
	all_friendly_builds = [[tuple([unit for unit in build[0] if unit[-1] == '0']),build[1]] for build in build_orders]
	
	all_opponent_builds = [[tuple([unit for unit in build[0] if unit[-1] == '1']),build[1]] for build in build_orders]

	# turn label from "True" to 1 and "False" to 0 and remove 0 or 1 at end of build order
	temp_all_friendly_builds = []
	for (build, result) in all_friendly_builds:
		if (result == True):
			tempResult = '1'
		else:
			tempResult = '0'
		
		#remove 0 or 1 at end of build order
		tempUnitOrBuilding = []
		for unitOrBuilding in build:
			tempUnitOrBuilding.append(unitOrBuilding[0:len(unitOrBuilding)-1]) 

		temp_all_friendly_builds.append((tempUnitOrBuilding, tempResult))
	all_friendly_builds = temp_all_friendly_builds

	# turn label from "True" to 1 and "False" to 0 and remove 0 or 1 at end of build order
	# note the tempResult has its values reversed because this says the opponent won
	temp_all_opponent_builds = []
	for (build, result) in all_opponent_builds:
		if (result == True):
			tempResult = '0'
		else:
			tempResult = '1'
		
		#remove 0 or 1 at end of build order
		tempUnitOrBuilding = []
		for unitOrBuilding in build:
			tempUnitOrBuilding.append(unitOrBuilding[0:len(unitOrBuilding)-1]) 

		temp_all_opponent_builds.append((tempUnitOrBuilding, tempResult))
	all_opponent_builds = temp_all_opponent_builds


	# make dataset of build A, build B, label 
	# the label will be 1 if A won, and 0 if B won
	datasetA = []
	datasetB = []
	labels = []
	for (friendly, opponent) in izip(all_friendly_builds, all_opponent_builds):
		(buildA, resultA) = friendly
		(buildB, resultB) = opponent 
		datasetA.append(' '.join(buildA) + ' .\n')
		datasetB.append(' '.join(buildB) + ' .\n')
		labels.append(resultA)

		# the dataset will also have a second entry that is the reverse
		# i.e. build B, build A, label (opposite of original label)
		# that it way the model doesn't overfit to picking build A over build B (or B over A)
	#	datasetA.append(' '.join(buildB) + ' .\n')
	#	datasetB.append(' '.join(buildA) + ' .\n')
	#	labels.append(resultB)

	# TODO
	# idk if we should do the above thing
	

	# split data in train, dev, test splits
	# split into .80 train and .20 temporary
	datasetATrain, datasetATemp, labelsTrain, labelsTemp = train_test_split(datasetA, labels, test_size = .2, shuffle = False)
	# split temporary into half dev and half test 
	datasetAdev, datasetAtest, labelsdev, labelsTest = train_test_split(datasetATemp, labelsTemp, test_size = .5, shuffle = False)

	# split into .80 train and .20 temporary
	datasetBTrain, datasetBTemp, labelsTrain, labelsTemp = train_test_split(datasetB, labels, test_size = .2, shuffle = False)
	# split temporary into half dev and half test 
	datasetBdev, datasetBtest, labelsdev, labelsTest = train_test_split(datasetBTemp, labelsTemp, test_size = .5, shuffle = False)


	print(datasetATrain[0])
	print(datasetAdev[0])
	print(datasetAtest[0])
	print(datasetBTrain[0])
	print(datasetBdev[0])
	print(datasetBtest[0])
	print(labelsTrain[0])
	print(labelsdev[0])
	print(labelsTest[0])

	# write dataset to files
	with open('s1.train', 'w') as datasetATrainFile:
		for element in datasetATrain:
			datasetATrainFile.write(element)

	with open('s2.train', 'w') as datasetBTrainFile:
		for element in datasetBTrain:
			datasetBTrainFile.write(element)

	with open('labels.train', 'w') as labelsTrainFile:
		for element in labelsTrain:
			labelsTrainFile.write(element + '\n')

	with open('s1.dev', 'w') as datasetAdevFile:
		for element in datasetAdev:
			datasetAdevFile.write(element)

	with open('s2.dev', 'w') as datasetBdevFile:
		for element in datasetBdev:
			datasetBdevFile.write(element)

	with open('labels.dev', 'w') as labelsdevFile:
		for element in labelsdev:
			labelsdevFile.write(element + '\n')

	with open('s1.test', 'w') as datasetAtestFile:
		for element in datasetAtest:
			datasetAtestFile.write(element)

	with open('s2.test', 'w') as datasetBtestFile:
		for element in datasetBtest:
			datasetBtestFile.write(element)

	with open('labels.test', 'w') as labelsTestFile:
		for element in labelsTest:
			labelsTestFile.write(element + '\n')


	# make our own word vectors
	X = []
	for element in datasetATrain:
		X.append(element.split())
	for element in datasetAdev:
		X.append(element.split())
	for element in datasetAdev:
		X.append(element.split())

	
	# train custom word2vec embeddings on the training set
	model = Word2Vec(X, size=300, window=5, min_count=5, workers=multiprocessing.cpu_count())
	model.wv.save_word2vec_format('buildOrderEmbeddings.300D.txt', binary=False)
	w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
	print(w2v['Hatchery'])	

if __name__ == '__main__':
	build_dataset()
