# Fish-Species-Classification-Using-K-Nearest-Neighbors 

This is the result of an artificial intelligence team project that predicts fish species using length and weight data with the K-Nearest Neighbors (KNN) classification algorithm.

## Motivation

Our team initiated this project to apply a machine learning classification algorithm to a simple structured dataset. While exploring possible datasets, we selected a fish dataset because it contained multiple fish species and numerical features that could be used for classification.

Using the Fish.csv dataset, we aimed to predict fish species based on two numerical features: `Weight` and `Length2`. We selected the K-Nearest Neighbors (KNN) algorithm because it is intuitive, simple to implement, and suitable for classification problems with relatively low data complexity.

Through this project, we experienced the basic machine learning workflow, including dataset preparation, feature selection, data splitting, standardization, model training, prediction, evaluation, and visualization of nearest neighbors.

## End-to-End Process (with Output)

## 1. Project Objective </br>

- Predict fish species using numerical length and weight data. </br>
- Apply the K-Nearest Neighbors (KNN) algorithm to a classification problem. </br>
- Compare the results of the first model using seven fish species and the second model using three selected fish species. </br>
- Visualize new prediction points and their nearest neighbors to understand the model behavior. </br>

## 2. Algorithm Selection: K-Nearest Neighbors </br>

- Used K-Nearest Neighbors (KNN) as the main classification algorithm. </br>
- KNN classifies a new data point based on the majority class among its nearest `k` neighbors. </br>
- The basic process of KNN is: </br>
  - Measure the distance between data points, usually using Euclidean distance. </br>
  - Select the nearest `k` neighbors. </br>
  - Assign the most frequent class among the neighbors as the prediction result. </br>

- KNN was selected because the project used a relatively simple numerical dataset, and KNN is easy to understand and suitable for basic classification tasks. </br>

## 3. Dataset Preparation </br>

- Loaded the fish dataset using `pd.read_csv()`. </br>
- The dataset was loaded from the following source: </br>
  - `https://raw.githubusercontent.com/satishgunjal/datasets/master/Fish.csv` </br>

- The original dataset included seven fish species: </br>
  - `Bream` </br>
  - `Roach` </br>
  - `Whitefish` </br>
  - `Parkki` </br>
  - `Perch` </br>
  - `Pike` </br>
  - `Smelt` </br>

- The dataset included several numerical features: </br>
  - `Weight` </br>
  - `Length1` </br>
  - `Length2` </br>
  - `Length3` </br>
  - `Height` </br>
  - `Width` </br>

- For simplicity, this project used only two features: </br>
  - `Weight` </br>
  - `Length2` </br>

- `Length2` represents the standard length of the fish. </br>

## 4. First Model Implementation: Seven Fish Species </br>

### 4.1 Feature and Target Construction </br>

- Extracted `Length2` and `Weight` from the dataset. </br>
- Combined the two features into `fish_data`. </br>
- Created `fish_target` manually to represent the fish species labels. </br>

```python
length = df.iloc[1:160, 3].astype(float)
weight = df.iloc[1:160, 1].astype(float)

fish_data = [[l, w] for l, w in zip(length, weight)]

fish_target = [0]*35 + [1]*20 + [2]*6 + [3]*11 + [4]*67 + [5]*17 + [6]*14
````

* The target labels were defined as: </br>

  * `Bream`: 0 </br>
  * `Roach`: 1 </br>
  * `Whitefish`: 2 </br>
  * `Parkki`: 3 </br>
  * `Perch`: 4 </br>
  * `Pike`: 5 </br>
  * `Smelt`: 6 </br>

### 4.2 Train-Test Split </br>

* Converted the feature and target data into NumPy arrays. </br>
* Created an index array using `np.arange()`. </br>
* Shuffled the index using `np.random.shuffle(index)`. </br>
* Used the shuffled index to manually split the data into training and testing sets. </br>

```python
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

index = np.arange(len(input_arr))

np.random.seed(42)
np.random.shuffle(index)

train_input = input_arr[index[:120]]
train_target = target_arr[index[:120]]
test_input = input_arr[index[120:]]
test_target = target_arr[index[120:]]
```

* The first 120 shuffled samples were used for training. </br>
* The remaining samples were used for testing. </br>

### 4.3 Feature Standardization </br>

* Calculated the mean and standard deviation of the training data. </br>
* Standardized both training and testing data using the training mean and standard deviation. </br>

```python
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std
```

* Standardization was applied because KNN is distance-based, and feature scale can affect nearest-neighbor calculation. </br>

### 4.4 Model Training and Prediction </br>

* Created a KNN classifier using `KNeighborsClassifier`. </br>
* Set `n_neighbors=6` for the first model. </br>
* Trained the model using the standardized training data. </br>

```python
kn = KNeighborsClassifier(n_neighbors=6)

kn.fit(train_scaled, train_target)
```

* Tested the model using three manually created input examples: </br>

  * Bream-like input: `[40, 1000]` </br>
  * Smelt-like input: `[9, 6]` </br>
  * Roach-like input: `[23, 160]` </br>

```python
new_scaled1 = ([40, 1000] - mean) / std # bream data
new_scaled2 = ([9, 6] - mean) / std # smelt data
new_scaled3 = ([23, 160] - mean) / std # roach data

print("Prediction 1:", kn.predict([new_scaled1]))
print("Prediction 2:", kn.predict([new_scaled2]))
print("Prediction 3:", kn.predict([new_scaled3]))

print("Accuracy:", kn.score(test_scaled, test_target))
```

### 4.5 First Model Result </br>

* The first model correctly predicted the Bream-like and Roach-like input data. </br>
* However, the Smelt-like input was incorrectly predicted as Pike. </br>
* The test accuracy was slightly above 60%. </br>

#### First Model Output </br>

<!-- Add your first model output screenshot here -->

<img width="" height="" alt="First model output" src="" /> </br>

## 5. Result Analysis of the First Model </br>

* Used `kn.kneighbors()` to identify the nearest neighbors of the new prediction inputs. </br>
* Visualized the training data, new input points, and nearest neighbors using `matplotlib`. </br>

```python
distances1, indexes1 = kn.kneighbors([new_scaled1])
distances2, indexes2 = kn.kneighbors([new_scaled2])
distances3, indexes3 = kn.kneighbors([new_scaled3])
```

* The visualization showed that the data points were densely distributed. </br>
* Because KNN depends on distance between data points, dense or overlapping data distributions can make classification more difficult. </br>
* This helped explain why the Smelt-like input was misclassified in the first model. </br>

#### First Model Visualization </br>

<!-- Add your first visualization screenshot here -->

<img width="" height="" alt="First model visualization" src="" /> </br>

## 6. Second Model Implementation: Three Fish Species </br>

### 6.1 Species Selection </br>

* To improve classification performance, the second model used only three fish species: </br>

  * `Bream` </br>
  * `Roach` </br>
  * `Smelt` </br>

* The selected features were still: </br>

  * `Length2` </br>
  * `Weight` </br>

```python
bream_length = df.iloc[2:36, 3].astype(float)
smelt_length = df.iloc[146:160, 3].astype(float)
roach_length = df.iloc[36:56, 3].astype(float)

bream_weight = df.iloc[2:36, 1].astype(float)
smelt_weight = df.iloc[146:160, 1].astype(float)
roach_weight = df.iloc[36:56, 1].astype(float)
```

### 6.2 Feature and Target Reconstruction </br>

* Combined the selected fish data into a new dataset. </br>
* Reconstructed the target labels for the three selected fish species. </br>

```python
length = bream_length.tolist() + roach_length.tolist() + smelt_length.tolist()
weight = bream_weight.tolist() + roach_weight.tolist() + smelt_weight.tolist()

fish_data = [[l, w] for l, w in zip(length, weight)]

fish_target = [0]*35 + [1]*20 + [6]*14
```

* The target labels were: </br>

  * `Bream`: 0 </br>
  * `Roach`: 1 </br>
  * `Smelt`: 6 </br>

### 6.3 Train-Test Split and Standardization </br>

* Used `np.random.shuffle(index)` again to randomly shuffle the dataset index. </br>
* Used 50 samples for training and the remaining samples for testing. </br>
* Applied standardization using the training data mean and standard deviation. </br>

```python
np.random.seed(42)
index = np.arange(len(input_arr))

np.random.shuffle(index)

train_input = input_arr[index[:50]]
train_target = target_arr[index[:50]]
test_input = input_arr[index[50:]]
test_target = target_arr[index[50:]]

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std
```

### 6.4 Model Training and Prediction </br>

* Created a new KNN classifier. </br>
* Set `n_neighbors=3` for the second model. </br>
* Trained the model using the standardized training data. </br>

```python
kn = KNeighborsClassifier(n_neighbors=3)

kn.fit(train_scaled, train_target)
```

* Tested the model using the same manually created examples: </br>

  * Bream-like input: `[40, 1000]` </br>
  * Smelt-like input: `[9, 6]` </br>
  * Roach-like input: `[23, 160]` </br>

```python
new_scaled1 = ([40, 1000] - mean) / std # bream -> 0
new_scaled2 = ([9, 6] - mean) / std # smelt -> 6
new_scaled3 = ([23, 160] - mean) / std # roach -> 1

print("Prediction 1:", kn.predict([new_scaled1]))
print("Prediction 2:", kn.predict([new_scaled2]))
print("Prediction 3:", kn.predict([new_scaled3]))

print("Accuracy:", kn.score(test_scaled, test_target))
```

## 7. Second Model Result </br>

* The second model correctly predicted all three manually created input examples. </br>
* The Bream-like input was predicted as Bream. </br>
* The Smelt-like input was predicted as Smelt. </br>
* The Roach-like input was predicted as Roach. </br>
* The test set accuracy was clearly improved compared with the first model. </br>

#### Second Model Output </br>

<!-- Add your second model output screenshot here -->

<img width="" height="" alt="Second model output" src="" /> </br>

## 8. Visualization of Nearest Neighbors </br>

* Visualized the standardized training data and the new prediction inputs. </br>
* Marked the nearest neighbors for each new input. </br>
* Generated separate visualizations for: </br>

  * Bream prediction </br>
  * Smelt prediction </br>
  * Roach prediction </br>

```python
plt.scatter(train_scaled[:, 0], train_scaled[:, 1], color='gray', label='Training data', alpha=0.6)

plt.scatter(new_scaled1[0], new_scaled1[1], color='red', marker='o', s=100, label='Bream (new)')
plt.scatter(new_scaled2[0], new_scaled2[1], color='blue', marker='o', s=100, label='Smelt (new)')
plt.scatter(new_scaled3[0], new_scaled3[1], color='green', marker='o', s=100, label='Roach (new)')

plt.xlabel('Length (standardized)')
plt.ylabel('Weight (standardized)')
plt.title('Visualization of Fish Data - New Predictions and Their Nearest Neighbors')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
```

#### Overall Visualization </br>

<!-- Add your overall visualization screenshot here -->

<img width="" height="" alt="Overall visualization" src="" /> </br>

#### Bream Prediction Visualization </br>

<!-- Add your Bream visualization screenshot here -->

<img width="" height="" alt="Bream prediction visualization" src="" /> </br>

#### Smelt Prediction Visualization </br>

<!-- Add your Smelt visualization screenshot here -->

<img width="" height="" alt="Smelt prediction visualization" src="" /> </br>

#### Roach Prediction Visualization </br>

<!-- Add your Roach visualization screenshot here -->

<img width="" height="" alt="Roach prediction visualization" src="" /> </br>

## 9. Model Evaluation </br>

* Used `kn.score()` to evaluate the test set accuracy. </br>
* Compared the first model and the second model based on prediction results and test accuracy. </br>

### First Model </br>

* Used seven fish species. </br>
* Used `n_neighbors=6`. </br>
* Correctly predicted Bream-like and Roach-like inputs. </br>
* Misclassified the Smelt-like input as Pike. </br>
* Test accuracy was slightly above 60%. </br>

### Second Model </br>

* Used three fish species: Bream, Roach, and Smelt. </br>
* Used `n_neighbors=3`. </br>
* Correctly predicted all three manually created input examples. </br>
* Test accuracy clearly improved compared with the first model. </br>

## 10. Learning Experience </br>

Through this project, we learned how a basic classification model can be implemented using Python and scikit-learn. We also learned that KNN performance can be affected by feature selection, data distribution, class overlap, standardization, and the value of `k`.

This project helped us understand the importance of checking model results visually, not only through accuracy scores but also through nearest-neighbor visualization.

### Difficulties </br>

* The first model used seven fish species, and some species were difficult to separate using only `Weight` and `Length2`. </br>
* The Smelt-like input was misclassified as Pike in the first model. </br>
* The data points were densely distributed, which made KNN classification more difficult. </br>
* The value of `k` affected the prediction result. </br>

### Solutions </br>

* Reduced the classification task from seven fish species to three selected fish species. </br>
* Used standardization to make distance-based KNN classification more appropriate. </br>
* Changed the number of neighbors from `n_neighbors=6` to `n_neighbors=3`. </br>
* Visualized the nearest neighbors to analyze why predictions succeeded or failed. </br>

## 11. Open Source SW </br>

* Created a machine learning classification model using Python. </br>
* Used open-source Python libraries for data handling, numerical computation, machine learning, and visualization. </br>
* Used an open fish dataset from GitHub. </br>

### Main Libraries </br>

* `pandas` </br>
* `numpy` </br>
* `matplotlib` </br>
* `scikit-learn` </br>

### Main Functions and Methods </br>

* `pd.read_csv()` — Load the dataset. </br>
* `df.iloc[]` — Select specific rows and columns from the dataset. </br>
* `.astype(float)` — Convert selected data to float type. </br>
* `.tolist()` — Convert pandas Series data to Python lists. </br>
* `np.array()` — Convert feature and target data into NumPy arrays. </br>
* `np.arange()` — Create an index array. </br>
* `np.random.seed()` — Fix the random seed for reproducibility. </br>
* `np.random.shuffle()` — Shuffle the dataset index. </br>
* `np.mean()` — Calculate the mean of the training data. </br>
* `np.std()` — Calculate the standard deviation of the training data. </br>
* `KNeighborsClassifier()` — Create the KNN classification model. </br>
* `kn.fit()` — Train the KNN model. </br>
* `kn.predict()` — Predict fish species for new input data. </br>
* `kn.score()` — Evaluate test set accuracy. </br>
* `kn.kneighbors()` — Find nearest neighbors for new input data. </br>
* `plt.scatter()` — Visualize data points. </br>
* `plt.xlabel()` — Set the x-axis label. </br>
* `plt.ylabel()` — Set the y-axis label. </br>
* `plt.title()` — Set the plot title. </br>
* `plt.legend()` — Add a legend to the plot. </br>
* `plt.grid()` — Add grid lines to the plot. </br>
* `plt.show()` — Display the visualization. </br>

## 👥 Team Member

201834811 Park Jaehak </br>
201934219 Kim Joonhee </br>
202135520 Kim Seungsu </br>

## ✔️ Source

* [Fish.csv Dataset](https://raw.githubusercontent.com/satishgunjal/datasets/master/Fish.csv) </br>
* [K-Nearest Neighbors - Wikipedia](https://ko.wikipedia.org/wiki/K-%EC%B5%9C%EA%B7%BC%EC%A0%91_%EC%9D%B4%EC%9B%83_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98) </br>
* [scikit-learn KNeighborsClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) </br>
* [Machine Learning Study Reference](https://zara49.tistory.com/58) </br>
* [Machine Learning and Deep Learning Study Reference](https://velog.io/@kitebull316/%ED%98%BC%EC%9E%90-%EA%B3%B5%EB%B6%80%ED%95%98%EB%8A%94-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EB%94%A5%EB%9F%AC%EB%8B%9D-1-3) </br>
