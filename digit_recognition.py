import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn import svm


# =============================================================================
#  Predict Handwritten digits using :
#  1. GaussianNB
#  2. Support Vector Machine
# =============================================================================


digits = load_digits()


X = digits.data
y = digits.target
# Check all keys
print('Keys: {}'.format(digits.keys()))

# Show first row of data
print(X[:1])


# Plot digits using pyplot
fig, axes = plt.subplots(10, 10, figsize=(8,8), subplot_kw={'xticks':[],'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='green')



iso = Isomap(n_components=2)
iso.fit(X)
data_projected = iso.transform(X)
print('Isomap data projected: ', data_projected.shape)
# Plot Isomap projected data 
plt.figure()
plt.scatter(data_projected[:, 0], data_projected[:,1], c=y, edgecolor='none', 
            alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)

# Split Data for training and testing
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)


# =============================================================================
#  Predict using GaussianNB
# =============================================================================
model = GaussianNB()
model.fit(train_X,train_y)

# Get prediction on test data
prediction = model.predict(test_X)
print('GaussianNB Prediction: ', prediction)
# Check Accuracy
accuracy = accuracy_score(test_y, prediction)
print('GaussianNB Accuracy: ', accuracy) # GaussianNB Accuracy:  0.8333333333333334


# =============================================================================
#  Predict using SVM
# =============================================================================
clf = svm.SVC(kernel='linear')

# Train classifier
clf.fit(train_X, train_y)

# Predict test dataset
prediction = clf.predict(test_X)
print('Predictions: ', prediction)
# Check Score
score = clf.score(test_X, test_y)
print('Score: ', score) # Score:  0.9711111111111111