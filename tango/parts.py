import matplotlib.pyplot as plt
import random

def dump_arr(fn, arr):
	fp = open(fn, "w")

	for line in arr:
		fp.write("%s\n" % line)

	fp.close()

fp = open("TRUMPF_TruLaserCenter_Dataset_2018.csv", "r")

part_cnts = {}

for cnt, line in zip(range(200000), fp):
	if cnt != 0:
		part_name = line.split(",")[2]

		if part_name in part_cnts:
			part_cnts[part_name] += 1
		else:
			part_cnts[part_name] = 1

fp.close()

print("%d parts" % len(part_cnts))

part_cnts_arr = []
over_1000 = 0

for key, value in part_cnts.items():
	part_cnts_arr.append(value)

	if value >= 1000:
		print(key, value)
		over_1000 += 1

print("%d parts over 1000" % over_1000)

part_cnts_arr.sort(reverse = True)
# plt.plot(part_cnts_arr)

# plt.show()

# names = []

# for key, val in part_cnts.items():
# 	names.append(key)

# train_parts = random.sample(names, 420)

# rem = []

# for name in names:
# 	if name not in train_parts:
# 		rem.append(name)

# valid_parts = random.sample(rem, 140)

# test_parts = []

# for name in names:
# 	if (name not in train_parts) and (name not in valid_parts):
# 		test_parts.append(name)

# dump_arr("train.txt", train_parts)
# dump_arr("valid.txt", valid_parts)
# dump_arr("test.txt", test_parts)

# print("Train cnt = %d" % (sum([part_cnts[x] for x in train_parts])))
# print("Valid cnt = %d" % (sum([part_cnts[x] for x in valid_parts])))
# print("Test cnt = %d" % (sum([part_cnts[x] for x in test_parts])))