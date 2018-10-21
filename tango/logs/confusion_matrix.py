fpIn = open("log_model5.txt", "r")

total = 0
correct = 0

tp = 0
tn = 0
fp = 0
fn = 0

for line in fpIn:
	y_true = float(line.split()[0])
	y_pred = float(line.split()[1])

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

	total += 1

fpIn.close()

print("Total = %d, correct = %d => accuracy = %lf" % (total, correct, float(correct) / float(total)))

print("%d %d\n%d %d" % (tp, fp, fn, tn))