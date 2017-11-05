from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline


def svc_model(X_train, y_train, num_folds = 5):
	svc = make_pipeline(preprocessing.StandardScaler(), LinearSVC())
	# Check the training time for the SVC
	scores = cross_val_score(svc, X_train, y_train, cv=num_folds)
	return scores
