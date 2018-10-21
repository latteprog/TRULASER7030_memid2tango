import matplotlib.pyplot as plt
import numpy as np
import sys

def get_ratio(threshold):
	fp1 = open("log_model1.txt", "r")
	fp2 = open("log_model2.txt", "r")
	fp3 = open("log_model3.txt", "r")
	fp4 = open("log_model4.txt", "r")
	fp5 = open("log_model5.txt", "r")

	total = 0
	correct = 0

	tp = 0
	tn = 0
	fp = 0
	fn = 0

	p = 0
	n = 0

	for l1, l2, l3, l4, l5 in zip(fp1, fp2, fp3, fp4, fp5):
		y_true = float(l1.split()[0])

		ensm = ""

		if float(l1.split()[1]) >= threshold:
			ensm += "1"
		else:
			ensm += "0"

		if float(l2.split()[1]) >= threshold:
			ensm += "1"
		else:
			ensm += "0"

		if float(l3.split()[1]) >= threshold:
			ensm += "1"
		else:
			ensm += "0"

		if float(l4.split()[1]) >= threshold:
			ensm += "1"
		else:
			ensm += "0"

		if float(l5.split()[1]) >= threshold:
			ensm += "1"
		else:
			ensm += "0"

		if ensm.count("1") >= 3:
			y_pred = 1.0
		else:
			y_pred = 0.0

		if y_true == y_pred:
			correct += 1

			if y_pred == 1.0:
				tp += 1
			else:
				tn += 1
		else:
			if y_pred == 1.0:
				fp += 1
			else:
				fn += 1

		if y_true == 1.0:
			p += 1
		else:
			n += 1

		total += 1

	fp1.close()
	fp2.close()
	fp3.close()
	fp4.close()
	fp5.close()

	print("Total = %d, correct = %d => accuracy = %lf" % (total, correct, float(correct) / float(total)))

	return (tp / p, fp / n)

points = []

for threshold in np.linspace(0.000001, 1, 100, endpoint = False):
	points.append(get_ratio(threshold))

plt.scatter([x[1] for x in points], [x[0] for x in points])
plt.title("NN Ensemble ROC")
plt.ylabel("False Positive Rate")
plt.xlabel("True Positive Rate")

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.savefig("NN_Ensm_ROC.png", dpi = 300)
