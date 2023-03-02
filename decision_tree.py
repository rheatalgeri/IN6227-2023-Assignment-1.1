import csv
import random
import time
from sklearn.metrics import confusion_matrix

def load_data(filename):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

data = load_data('adult.csv')


def split_data(data, split_ratio): # split the data into train and test sets
    train_size = int(len(data) * split_ratio)
    train_set = []
    test_set = list(data)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
    return train_set, test_set

train_data, test_data = split_data(data, 0.7)

def gini_index(groups, classes):  # calculates  the gini index of a split
    num_of_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            probability = [row[-1] for row in group].count(class_val) / size
            score += probability * probability
        gini += (1.0 - score) * (size / num_of_instances)
    return gini

def test_split(index, value, dataset): # tests a split on a particular attribute
    left = []
    right = []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_split(dataset): # finds the best split based on the gini index
    classes = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, classes)
            if gini < best_score:
                best_index, best_value, best_score, b_groups = index, row[index], gini, groups
    return {'index':best_index, 'value':best_value, 'groups':b_groups}

def get_max_class(group): # returns the most common class
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth): # recursively splits a node of the decision tree until the stopping criterion is fulfilled
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = get_max_class(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = get_max_class(left), get_max_class(right)
        return
    if len(left) <= min_size:
        node['left'] = get_max_class(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = get_max_class(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

def build_tree(training_set, max_depth, min_size): # builds the tree using the training data
    root = get_split(training_set)
    split(root, max_depth, min_size, 1)
    return root


start_time = time.time()

tree = build_tree(train_data, 5, 10) # training the tree

end_time = time.time()

time_taken = end_time - start_time

print("time taken to train tree: ", time_taken)

def classify(node, row): # goes through the built decision tree to classify a new row, mostly on the test data
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return classify(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return classify(node['right'], row)
        else:
            return node['right']

def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# measure the time to make predictions on the test data
start_time = time.time()

predictions = []

for row in test_data:
    prediction = classify(tree, row)
    predictions.append(prediction)

actual = [row[-1] for row in test_data]
accuracy = calculate_accuracy(actual, predictions)

print('Accuracy: {:.2f}%'.format(accuracy))

end_time = time.time()
time_taken = end_time - start_time

print(f"Time to make predictions: {time_taken} seconds")

matrix = confusion_matrix(actual, predictions)

print('Confusion matrix:')
print(matrix)


print(type(matrix))
TP = matrix[0,0]
FP = matrix[0,1]
FN = matrix[1,0]
TN = matrix[1,1]

print("TP: ", TP)
print("FP: ", FP)
print("FN: ", FN)
print("TN: ", TN)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print("F1 Score:", f1_score)
print("Precision:", precision)
print("Recall:", recall)