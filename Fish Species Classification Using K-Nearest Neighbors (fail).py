import matplotlib.pyplot as plt
import sklearn.metrics as mt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = 'https://raw.githubusercontent.com/satishgunjal/datasets/master/Fish.csv'
df = pd.read_csv(data)

length = df.iloc[1:160, 3].astype(float)
weight = df.iloc[1:160, 1].astype(float)

fish_data = [[l, w] for l, w in zip(length, weight)]

# (Bream 0, Roach 1, Whitefish 2, Parkki 3, Perch 4, Pike 5, Smelt 6)
fish_target = [0]*35 + [1]*20 + [2]*6 + [3]*11 + [4]*67 + [5]*17 + [6]*14
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

index = np.arange(len(input_arr))

np.random.seed(42)
np.random.shuffle(index)

train_input = input_arr[index[:120]]
train_target = target_arr[index[:120]]
test_input = input_arr[index[120:]]
test_target = target_arr[index[120:]]

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std

new_scaled1 = ([40, 1000]- mean) / std # bream data
new_scaled2 = ([9, 6] - mean) / std # smelt data
new_scaled3 = ([23, 160] - mean) / std # roach data

kn = KNeighborsClassifier(n_neighbors=6)

kn.fit(train_scaled, train_target)

print("예측1:", kn.predict([new_scaled1]))
print("예측2:", kn.predict([new_scaled2]))
print("예측3:", kn.predict([new_scaled3]))

print("정확도:", kn.score(test_scaled, test_target))

distances1, indexes1 = kn.kneighbors([new_scaled1])
distances2, indexes2 = kn.kneighbors([new_scaled2])
distances3, indexes3 = kn.kneighbors([new_scaled3])

plt.figure(figsize=(14, 8))

plt.scatter(train_scaled[:, 0], train_scaled[:, 1], color='gray', label='Training data', alpha=0.6)

plt.scatter(new_scaled1[0], new_scaled1[1], color='red', marker='o', s=100, label='Bream (new)')
plt.scatter(new_scaled2[0], new_scaled2[1], color='blue', marker='o', s=100, label='Smelt (new)')
plt.scatter(new_scaled3[0], new_scaled3[1], color='green', marker='o', s=100, label='Roach (new)')

plt.scatter(train_scaled[indexes1, 0], train_scaled[indexes1, 1], color='red', marker='s', s=50, label='Nearest to Bream')
plt.scatter(train_scaled[indexes2, 0], train_scaled[indexes2, 1], color='blue', marker='s', s=50, label='Nearest to Smelt')
plt.scatter(train_scaled[indexes3, 0], train_scaled[indexes3, 1], color='green', marker='s', s=50, label='Nearest to Roach')

plt.xlabel('Length (standardized)')
plt.ylabel('Weight (standardized)')
plt.title('Visualization of Fish Data - New Predictions and Their Nearest Neighbors')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
