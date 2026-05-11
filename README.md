# Fish-Species-Classification-Using-K-Nearest-Neighbors 

This is the result of an artificial intelligence team project that predicts fish species using length and weight data with the K-Nearest Neighbors (KNN) classification algorithm.

## Motivation

Our team initiated this project to apply a machine learning classification algorithm to a simple numerical dataset. While exploring possible datasets, we selected a fish dataset because it contained multiple fish species and numerical features that could be used for classification.

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

* The first 120 shuffled samples were used for training. </br>
* The remaining samples were used for testing. </br>

### 4.3 Feature Standardization </br>

* Calculated the mean and standard deviation of the training data. </br>
* Standardized both training and testing data using the training mean and standard deviation. </br>

* Standardization was applied because KNN is distance-based, and feature scale can affect nearest-neighbor calculation. </br>

### 4.4 Model Training and Prediction </br>

* Created a KNN classifier using `KNeighborsClassifier`. </br>
* Set `n_neighbors=6` for the first model. </br>
* Trained the model using the standardized training data. </br>

* Tested the model using three manually created input examples: </br>

  * Bream-like input: `[40, 1000]` </br>
  * Smelt-like input: `[9, 6]` </br>
  * Roach-like input: `[23, 160]` </br>

### 4.5 First Model Result </br>

* The first model correctly predicted the Bream-like and Roach-like input data. </br>
* However, the Smelt-like input was incorrectly predicted as Pike. </br>
* The test accuracy was slightly above 60%. </br>

#### First Model Output </br>

<img width="171" height="67" alt="image" src="https://github.com/user-attachments/assets/dfc1c860-9795-4dec-830c-d7ff7d24e628" />

## 5. Result Analysis of the First Model </br>

* Used `kn.kneighbors()` to identify the nearest neighbors of the new prediction inputs. </br>
* Visualized the training data, new input points, and nearest neighbors using `matplotlib`. </br>

* The visualization showed that the data points were densely distributed. </br>
* Because KNN depends on distance between data points, dense or overlapping data distributions can make classification more difficult. </br>
* This helped explain why the Smelt-like input was misclassified in the first model. </br>

#### First Model Visualization </br>

<img width="468" height="244" alt="image" src="https://github.com/user-attachments/assets/4c84165c-10f2-42d3-9fa9-7a08f23cb725" /> </br>

## 6. Second Model Implementation: Three Fish Species </br>

### 6.1 Species Selection </br>

* To improve classification performance, the second model used only three fish species: </br>

  * `Bream` </br>
  * `Roach` </br>
  * `Smelt` </br>

* The selected features were still: </br>

  * `Length2` </br>
  * `Weight` </br>

### 6.2 Feature and Target Reconstruction </br>

* Combined the selected fish data into a new dataset. </br>
* Reconstructed the target labels for the three selected fish species. </br>

* The target labels were: </br>

  * `Bream`: 0 </br>
  * `Roach`: 1 </br>
  * `Smelt`: 6 </br>

### 6.3 Train-Test Split and Standardization </br>

* Used `np.random.shuffle(index)` again to randomly shuffle the dataset index. </br>
* Used 50 samples for training and the remaining samples for testing. </br>
* Applied standardization using the training data mean and standard deviation. </br>

### 6.4 Model Training and Prediction </br>

* Created a new KNN classifier. </br>
* Set `n_neighbors=3` for the second model. </br>
* Trained the model using the standardized training data. </br>

* Tested the model using the same manually created examples: </br>

  * Bream-like input: `[40, 1000]` </br>
  * Smelt-like input: `[9, 6]` </br>
  * Roach-like input: `[23, 160]` </br>

## 7. Second Model Result </br>

* The second model correctly predicted all three manually created input examples. </br>
* The Bream-like input was predicted as Bream. </br>
* The Smelt-like input was predicted as Smelt. </br>
* The Roach-like input was predicted as Roach. </br>
* The test set accuracy was clearly improved compared with the first model. </br>

#### Second Model Output </br>

<img width="229" height="78" alt="image" src="https://github.com/user-attachments/assets/a2b81830-8be6-4576-91d3-909818fbb5a4" /> </br>

## 8. Visualization of Nearest Neighbors </br>

* Visualized the standardized training data and the new prediction inputs. </br>
* Marked the nearest neighbors for each new input. </br>
* Generated separate visualizations for: </br>

  * Bream prediction </br>
  * Smelt prediction </br>
  * Roach prediction </br>

#### Overall Visualization </br>

<img width="291" height="151" alt="image" src="https://github.com/user-attachments/assets/6a8ea2ab-761c-440c-9383-38b13e0efb0b" /> </br>

#### Bream Prediction Visualization </br>

<img width="278" height="155" alt="image" src="https://github.com/user-attachments/assets/c0650244-9fd3-441b-9459-f647cf6c3549" /> </br>

#### Smelt Prediction Visualization </br>

<img width="267" height="145" alt="image" src="https://github.com/user-attachments/assets/71005a26-c9d8-4ad7-af87-ab9a1d2b5909" /> </br>

#### Roach Prediction Visualization </br>

<img width="278" height="146" alt="image" src="https://github.com/user-attachments/assets/035883d6-b13a-4e10-9566-d78eca4be767" /> </br>

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
