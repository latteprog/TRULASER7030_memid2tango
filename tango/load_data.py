# constants
idxs = [6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
means = [0.000007, 2.418281, 169170.419194, 0.256501, 0.306464, 86412.562560, 55485.612983, 42643.977893, 71316.274452, 1214.230962, 163414.750118, 11.050216, 0.936469, 85451.685641, 376.348199, 228.035609, 456.510692, 1132.019174, 814.558767, 167805.293004, 1.562458, 2.710059, 525.043414, 713.133830, 626.491283, 740.069335, 50089.095717, 2463.863517, 408.829926, 42064.752464, 0.403918, 0.890947, 27.570210, 26.674127, 79.136793, 240.191167, 241.907705, 80.093217, 0.727531, 0.730557, 233.419009, 12365.044902, 27.087590, 46723369745.922150, 0.526010, 5244.178118, 2624.405358, 17.310393, 39.660273, 124.443088, 123.599911, 40.124101, 13.729600, 13.188365, 0.206088, 0.538865, 0.536565, 0.209497, 0.074705, 0.071430, 0.206586, 34.243430, 137.548154, 127.617706, 58.438353, 9222.900550, 3272.712117, 3349.799823, 87.089568, 141.956909, 435.600459, 396.670453, 0.289334, 0.750146, -408.303916, 44.286620, 57.357353, 0.350906, 0.947088, 0.762087, 1.391638, 0.087788]
stds = [0.000002, 1.161493, 55293.037079, 0.097354, 0.108561, 54614.177442, 63103.151537, 66572.037688, 120988.978461, 796.458833, 292167.412631, 21.366096, 0.140448, 134990.555714, 284.315167, 206.559097, 300.142074, 727.214533, 558.077523, 261649.803316, 0.505347, 3.775921, 408.114000, 396.230868, 386.153093, 376.783766, 86611.224467, 2949.523511, 307.884282, 76697.457421, 0.263390, 0.946408, 25.809815, 27.018395, 62.772748, 155.915909, 158.077745, 62.584176, 0.445230, 0.443671, 220.051454, 19375.174528, 26.614973, 3262672558376.322754, 0.497069, 5874.140096, 1697.833268, 13.844940, 34.030471, 97.713311, 96.628506, 34.098606, 14.456724, 14.454057, 0.131276, 0.080672, 0.082897, 0.134094, 0.063660, 0.064576, 0.404855, 36.987943, 94.814347, 89.457048, 36.540196, 2677.291177, 4638.244402, 4586.920645, 186.707478, 222.817139, 361.895911, 269.018127, 0.401442, 1.376301, 307.873704, 49.329394, 54.329906, 0.201370, 0.543492, 0.475353, 0.868035, 0.485268]

special_handling_idxs = [3, 9, 87, 95, 98]

# returns specific feature vector for some attributes
def special_handler(s, idx):
	if idx == 95:
		val = [0.0] * 3

		if float(s) == -1.0:
			val[0] = 1.0
		elif float(s) == 0.0:
			val[1] = 1.0
		elif float(s) == 1.0:
			val[2] = 1.0

		return val
	elif idx == 87:
		vals = [0, 1, 10, 11, 12, 13, 18, 2, 20, 25, 26, 27, 28, 3, 4, 5, 6, 7, 8, 9]
		val = [0.0] * 20

		for i in range(20):
			if float(s) == vals[i]:
				val[i] = 1.0

		return val
	elif idx == 9:
		val = [0.0] * 4

		if s == "":
			val[0] = 1.0
		elif s == "O2":
			val[1] = 1.0
		elif s == "N2":
			val[2] = 1.0
		elif s == "AI":
			val[3] = 1.0

		return val
	elif idx == 3:
		val = [0.0] * 5

		if s == "A0280E0012":
			val[0] = 1.0
		elif s == "A0280E0010":
			val[1] = 1.0
		elif s == "A0280E0009":
			val[2] = 1.0
		elif s == "A0280E0006":
			val[3] = 1.0
		elif s == "A0280E0021":
			val[4] = 1.0

		return val
	elif idx == 98:
		vals = [60, 95, 100, 6, 8, 75, 4, 10, 0, 2, 90, 70, 40, 50, 30, 85, 80]

		val = [0.0] * 17

		for i in range(17):
			if float(s) == vals[i]:
				val[i] = 1.0

		return val

# reads array from file
def load_arr(fn):
	arr = []

	with open(fn, "r") as fp:
		for line in fp:
			arr.append(line.split()[0])

	return arr

# loads data and split into 3 sets by part id
def load_data(chosen_idxs):
	train_names = load_arr("train.txt")
	valid_names = load_arr("valid.txt")
	test_names = load_arr("test.txt")

	fp = open("TRUMPF_TruLaserCenter_Dataset_2018.csv", "r")

	train = []
	valid = []
	test = []

	idxs_inv = {}

	for key, val in enumerate(idxs):
		idxs_inv[val] = key

	for cnt, line in zip(range(200000), fp):
		if cnt != 0:
			val = []

			line_split = line.split(",")

			for chosen_idx in chosen_idxs:
				if chosen_idx in special_handling_idxs:
					val += special_handler(line_split[chosen_idx], chosen_idx)
				else:
					if chosen_idx not in idxs_inv:
						continue

					idx_inv = idxs_inv[chosen_idx]

					raw_val = float(line_split[chosen_idx])
					val.append((raw_val - means[idx_inv]) / stds[idx_inv])

			if line_split[2] in train_names:
				train.append((val, float(line[-2])))
			elif line_split[2] in valid_names:
				valid.append((val, float(line[-2])))
			elif line_split[2] in test_names:
				test.append((val, float(line[-2])))

	fp.close()

	return train, valid, test