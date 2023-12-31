import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.data[0], breast_cancer_data.feature_names)
print("break")

print(breast_cancer_data.target)
print(breast_cancer_data.target_names)
print("break")

from sklearn.model_selection import train_test_split

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)

print(len(training_data), len(training_labels))
print("break")
from sklearn.neighbors import KNeighborsClassifier
curr_best_k = 1
curr_best_score = 0
accuracies = []
for i in range(1, 101):
  classifier = KNeighborsClassifier(n_neighbors=i)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))
  """if (classifier.score(validation_data, validation_labels) > curr_best_score):
    curr_best_score = classifier.score(validation_data, validation_labels)
    curr_best_k = i
print(curr_best_k, curr_best_score)"""
import matplotlib.pyplot as plt
k_list = list(range(1, 101))
print(k_list[:5])
plt.plot(k_list, accuracies)
plt.show()
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
