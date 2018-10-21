import random
import tensorflow as tf

import load_data
import Layers

def runOptimize(epochs):
	global train, valid
	global model_x, model_y_true, model_y_pred, model_loss, model_opt, model_sess, model_precision, model_recall

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

		print("Training loss = %lf" % (iterationLoss / float(len(train))),)
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

		print("Validation loss = %lf" % (iterationLoss / float(len(valid))),)
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

train, valid, test = load_data.load_data([9, 11, 50, 49, 10, 60, 57, 58, 59, 82, 12, 7, 90, 64, 53, 25, 13, 15, 63, 83, 34, 38, 46, 43, 36, 91, 92, 14, 81, 85, 80, 84, 55, 93, 94, 42, 20, 24, 45, 44, 54, 8])
# train, valid, test = load_data.load_data([10, 18, 24, 44, 45, 51, 53, 55, 68, 72, 80, 86])
# train, valid, test = load_data.load_data([11, 50, 49, 10, 60, 57, 58, 59, 82, 12, 7, 90, 64, 53, 25, 13, 15, 63, 83, 34])
# train, valid, test = load_data.load_data([3, 5, 9, 11, 50, 49, 10, 60, 57, 58, 59, 82, 12, 7, 90, 64, 53, 25, 13, 15, 63, 83, 34, 38, 46, 43, 36, 91, 92, 14, 81, 85, 80, 84, 55, 93, 94, 42, 20, 24, 45, 44, 54, 8, 6, 41, 74, 23, 22, 47, 48, 33, 27, 26, 73, 51, 16, 95, 35, 88, 76, 37, 68, 17, 65, 86, 18, 21, 72, 31, 71, 66, 79, 67, 29, 77, 78, 89, 28, 30, 87, 32, 19, 52, 56])

train = train + valid

model_x = tf.placeholder(tf.float32, shape = [None, 45])
model_y_true = tf.placeholder(tf.float32, shape = [None, 1])
model_hidden_1, _ = Layers.fullyConnected(model_x, name = "hidden_1", output_size = 8, activation_fn = "relu")
model_hidden_2, _ = Layers.fullyConnected(model_hidden_1, name = "hidden_2", output_size = 4, activation_fn = "relu")
model_y_pred, _ = Layers.fullyConnected(model_hidden_2, name = "y_pred", output_size = 1, activation_fn = "sigmoid")

model_loss = tf.reduce_mean(tf.pow(model_y_pred - model_y_true, 2))
model_opt = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(model_loss)

model_y_pred_discrete = tf.cast(tf.greater(model_y_pred, 0.5), dtype = tf.float32)
model_accuracy = tf.reduce_mean(tf.cast(tf.equal(model_y_true, model_y_pred_discrete), dtype = tf.float32))

model_precision = tf.metrics.precision(model_y_true, model_y_pred_discrete)
model_recall = tf.metrics.recall(model_y_true, model_y_pred_discrete)

model_sess = tf.Session()
model_sess.run(tf.global_variables_initializer())

for epoch in range(10):
	runOptimize(1)
	runTest()