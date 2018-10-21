import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import load_data

# data = np.array([x[0] for x in load_data.load_data([11, 50, 49, 10, 60, 57, 58, 59, 82, 12, 7, 90, 64, 53, 25, 13, 15, 63, 83, 34, 38, 46, 43, 36, 91, 92, 14, 81, 85, 80, 84, 55, 93, 94, 42, 20, 24, 45, 44, 54, 8, 6, 41, 74, 23, 22, 47, 48, 33, 27, 26, 73, 51, 16, 95, 35, 88, 76, 37, 68, 17, 65, 86, 18, 21, 72, 31, 71, 66, 79, 67, 29, 77, 78, 89, 28, 30, 87, 32, 19, 52, 56])])
# data = np.array([x[0] for x in load_data.load_data(list(range(15, 33)))])

raw, _, _ = load_data.load_data(list(range(33, 96)))

data = np.array([x[0] for x in raw])

print("Loaded")

pca = PCA(n_components = 5)
pca.fit(data)

print(pca.explained_variance_ratio_)
print(pca.get_params())

fitted = pca.transform(data)

fp = open("id03_pca'd.txt", "w")

for x in fitted:
	fp.write("%lf %lf %lf %lf %lf\n" % (x[0], x[1], x[2], x[3], x[4]))

fp.close()

# for cnt, x, y in zip(range(200000), fitted, raw):
# 	if cnt % 50 == 0:
# 		plt.scatter(x[0], x[1], color = ["#FF0000", "#00FF00"][int(y[1])])

# plt.show()