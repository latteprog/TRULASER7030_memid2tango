import numpy as np
import random
import tensorflow as tf

import load_data
import Layers

def runOptimize(epochs):
	global train, valid
	global model_x, model_y_true, model_y_pred, model_loss, model_opt, model_sess

	for epoch in range(epochs):
		# train
		random.shuffle(train)
		iterationLoss = 0.0
		accuracy = 0.0

		for i in range(0, len(train), 64):
			lowerIdx = i
			upperIdx = min(i + 64, len(train))

			feedDict = { model_x: [x[0] for x in train[lowerIdx: upperIdx]], model_y_true: [[x[1]] for x in train[lowerIdx: upperIdx]] }

			[_, loss, acc] = model_sess.run([model_opt, model_loss, model_accuracy], feed_dict = feedDict)

			iterationLoss += (upperIdx - lowerIdx) * loss
			accuracy += (upperIdx - lowerIdx) * acc

		print("Training loss = %lf" % (iterationLoss / float(len(train))))
		print("Training accuracy = %lf" % (accuracy / float(len(train))))

		# valid
		iterationLoss = 0.0
		accuracy = 0.0

		for i in range(0, len(valid), 64):
			lowerIdx = i
			upperIdx = min(i + 64, len(valid))

			feedDict = { model_x: [x[0] for x in valid[lowerIdx: upperIdx]], model_y_true: [[x[1]] for x in valid[lowerIdx: upperIdx]] }
			[loss, acc] = model_sess.run([model_loss, model_accuracy], feed_dict = feedDict)

			iterationLoss += (upperIdx - lowerIdx) * loss
			accuracy += (upperIdx - lowerIdx) * acc

		print("Validation loss = %lf" % (iterationLoss / float(len(valid))))
		print("Validation accuracy = %lf" % (accuracy / float(len(valid))))

def runTest():
	global test
	global model_x, model_y_true, model_y_pred, model_loss, model_opt, model_sess

	# test
	iterationLoss = 0.0
	accuracy = 0.0

	for i in range(0, len(test), 64):
		lowerIdx = i
		upperIdx = min(i + 64, len(test))

		feedDict = { model_x: [x[0] for x in test[lowerIdx: upperIdx]], model_y_true: [[x[1]] for x in test[lowerIdx: upperIdx]] }
		[loss, acc] = model_sess.run([model_loss, model_accuracy], feed_dict = feedDict)

		iterationLoss += (upperIdx - lowerIdx) * loss
		accuracy += (upperIdx - lowerIdx) * acc

	print("Testing loss = %lf" % (iterationLoss / float(len(test))))
	print("Testing accuracy = %lf" % (accuracy / float(len(test))))

train = []
valid = []
test = []

# load
with open("pca_data.txt", "r") as fp:
	for line in fp:
		line_split = line.split()

		sample = [[float(x) for x in line_split[1: -1]], float(line_split[-1])]

		if line_split[0] == "train":
			train.append(sample)
		elif line_split[0] == "valid":
			valid.append(sample)
		elif line_split[0] == "test":
			test.append(sample)

# mean/stddev normalise
for i in range(19):
	mean = np.mean([x[0][i] for x in train])
	std = np.std([x[0][i] for x in train])

	for j in range(len(train)):
		train[j][0][i] = (train[j][0][i] - mean) / std

	for j in range(len(valid)):
		valid[j][0][i] = (valid[j][0][i] - mean) / std

	for j in range(len(test)):
		test[j][0][i] = (test[j][0][i] - mean) / std

# model
model_x = tf.placeholder(tf.float32, shape = [None, 19])
model_y_true = tf.placeholder(tf.float32, shape = [None, 1])
model_hidden_1, _ = Layers.fullyConnected(model_x, name = "hidden_1", output_size = 8, activation_fn = "relu")
model_hidden_2, _ = Layers.fullyConnected(model_hidden_1, name = "hidden_2", output_size = 4, activation_fn = "relu")
model_y_pred, _ = Layers.fullyConnected(model_hidden_2, name = "y_pred", output_size = 1, activation_fn = "sigmoid")

model_loss = tf.reduce_mean(tf.pow(model_y_pred - model_y_true, 2))
model_opt = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(model_loss)

model_y_pred_discrete = tf.cast(tf.greater(model_y_pred, 0.5), dtype = tf.float32)
model_accuracy = tf.reduce_mean(tf.cast(tf.equal(model_y_true, model_y_pred_discrete), dtype = tf.float32))

model_sess = tf.Session()
model_saver = tf.train.Saver(max_to_keep = None)

model_sess.run(tf.global_variables_initializer())

# run
for epoch in range(10):
	runOptimize(1)
	runTest()

# model_saver.save(sess = model_sess, save_path = "pca_classifier/model5")