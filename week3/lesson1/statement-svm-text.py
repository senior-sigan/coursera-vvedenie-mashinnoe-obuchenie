import numpy as np
from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space'])

X = newsgroups.data
y = newsgroups.target
vectorizer = TfidfVectorizer()
X_scored = vectorizer.fit_transform(X, y)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X_scored, y)
for a in gs.grid_scores_:
    print(a.mean_validation_score)
    print(a.parameters)

C = 1.0

clf_good = SVC(C=C, kernel='linear', random_state=241)
clf_good.fit(X_scored, y)
weights = clf_good.coef_
print(weights)

topWeights = sorted(zip(weights.indices, weights.data), key=lambda e: abs(e[1]), reverse=True)[:10]
feature_mapping = vectorizer.get_feature_names()
words = [feature_mapping[i] for (i, w) in topWeights]
print(words)
