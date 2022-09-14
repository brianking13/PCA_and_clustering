import numpy as np
from matplotlib import pyplot as plt
from csv import reader
from sklearn.neighbors import NearestCentroid


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    dataset = [[float(i) for i in j] for j in dataset]
    return dataset


def convert_to_image(data):
    image_data = np.array(data[0:64])
    final_image = image_data.reshape(8, 8)
    return final_image


def mean(data):
    mean = np.empty([np.shape(data)[1]])
    for i in range(0, np.shape(data)[1]):
        for j in range(0, np.shape(data)[0]):
            mean[i] = mean[i] + (data[j][i])
        mean[i] = mean[i] / np.shape(data)[0]
    return mean


def split_labels(data):
    remove_label = []
    train_labels = []
    for i in range(0, len(data)):
        remove_label.append(data[i][0:64])
        train_labels.append(data[i][64])
    training_data_no_label = np.array(remove_label)
    return training_data_no_label, train_labels


def nearest_mean_classifier(X_train, y_train, X_test, y_test):
    model = NearestCentroid()
    model.fit(X_train, y_train)
    return model.score(X_test, y_test) * 100


def prediction_accuracy(training_data_no_label, train_labels, testing_data_no_label, test_labels, number):
    classifier_data = []
    classifier_data_labels = []
    for i in range(0, len(test_labels)):
        if test_labels[i] == number:
            classifier_data.append(testing_data_no_label[i])
            classifier_data_labels.append(test_labels[i])
    return nearest_mean_classifier(training_data_no_label, train_labels, classifier_data, classifier_data_labels)


# Import test data
test_data = load_csv('optdigits.tes')
training_data = load_csv('optdigits.tra')


# Plot first 15 numbers
plt.figure(figsize=(8, 8))
for i in range(1,16):
    plt.subplot(3, 5, i)
    plt.imshow(convert_to_image(training_data[i-1]), cmap='Blues')
plt.show()


# Create numpy arrays with all data points
training_data_no_label, train_labels = split_labels(training_data)
testing_data_no_label, test_labels = split_labels(test_data)

mu = mean(training_data_no_label)

# Find Eigenvectors
covariance_matrix = np.cov(np.transpose(training_data_no_label))
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
eigenvectors = np.transpose(eigenvectors)


# # Scree Plot
# plt.figure(2)
# plt.plot(eigenvalues)
# plt.xlabel("Eigenvectors")
# plt.ylabel("Eigenvalues")
# plt.show()


# Plot new 2D Space
W=eigenvectors[0:2]
# for i in range(0,1):
#     np.append(W,eigenvectors[i],axis=1)
projection = np.matmul(W, np.transpose(training_data_no_label-mu))
x_ax = projection[0]
y_ax = projection[1]
plt.figure(3)
plt.scatter(x_ax,y_ax,s=1, color="cyan")


# Annotate first 100 points
for i in range(0,100):
    plt.annotate(int(training_data[i][64]), (x_ax[i], y_ax[i]), fontsize=9)
plt.show()


# Calculate Reconstruction Error
mu_proj  = mean(np.transpose(projection))
mu_proj = mu_proj.reshape(2,1)
reverse_projection = np.matmul(np.transpose(W), projection+mu_proj)
error = 0
tdnl = np.transpose(training_data_no_label)
for i in range(0,np.shape(projection)[0]):
    for j in range(0,np.shape(projection)[1]):
        change = (tdnl[i][j]-reverse_projection[i][j])**2
        error = error + change
print("Reconstruction Error: " + str(error))

reverse_projection2 = np.transpose(reverse_projection)
# Show reconstructed data
plt.figure(figsize=(8, 8))
for i in range(1,16):
    plt.subplot(3, 5, i)
    plt.imshow(convert_to_image(reverse_projection2[i-1]), cmap='Blues')
plt.show()


# Nearest mean classifier for all dimensions
print("Using all 64 Dimensions:")
for i in range(0,10):
    score = prediction_accuracy(training_data_no_label, train_labels, testing_data_no_label, test_labels, i)
    print("Accuracy for number " + str(i) + " is " + str(score) + "%")


# Nearest mean classifier for 2D space
projection_test =np.matmul(eigenvectors,np.transpose(testing_data_no_label))
print("\nUsing feature extraction: ")
for i in range(0,10):
    score = prediction_accuracy(np.transpose(np.array([projection[0], projection[1]])), train_labels, np.transpose(np.array([projection_test[0], projection_test[1]])), test_labels, i)
    print("Accuracy for number " + str(i) + " is " + str(score) + "%")














