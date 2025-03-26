from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report


class ModelSelection:
    def __init__(self, df, feature):
        self.df = df
        self.X = df.drop(feature, axis=1)
        self.y = df[feature]
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.scaler = StandardScaler()
        self.normaliser = MinMaxScaler()

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_test, self.y_test, test_size=0.5, random_state=42)

        return self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val

    def feature_standardiser(self):

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.X_val = self.scaler.transform(self.X_val)

        return self.X_train, self.X_test, self.X_val

    def feature_normaliser(self):

        self.X_train, self.X_test, self.X_val = self.feature_standardiser()

        self.X_train = self.normaliser.fit_transform(self.X_train)
        self.X_test = self.normaliser.transform(self.X_test)
        self.X_val = self.normaliser.transform(self.X_val)

        return self.X_train, self.X_test, self.X_val

    def classifier(self):
        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val = self.split_data()
        norm_X_train, norm_X_test, norm_X_val = self.feature_normaliser()

        estimators = [LogisticRegression({'class_weight': ['balansed']}), KNeighborsClassifier(),
                      RandomForestClassifier({'class_weight': ['balansed']})]
        param_grids = [{'C': [0.01, 0.1, 1, 10, 100], 'solver': ['saga'], 'max_iter': [10000], 'penalty': ['elasticnet'],
                        'l1_ratio': [0.01, 0.1, 0.5, 1], 'dual': [False]},
                       {'n_neighbors': [1, 50, 100, 150, 175], 'metric': ['euclidean', 'manhattan', 'minkowski'],
                        'weights': ['uniform', 'distance'], 'algorithm': ['auto']},
                       {'max_depth': [None], 'criterion': ['gini'], 'n_estimators': [50], 'max_features': ['log2'], 'max_leaf_nodes': [3]}]

        for i, (estimator, param_grid) in enumerate(zip(estimators, param_grids)):
            clf = GridSearchCV(estimator, param_grid,
                               cv=5, n_jobs=-1, verbose=0, scoring='recall')
            clf.fit(norm_X_train, self.y_train)
            print(
                f"Best parameters for {estimator}: {clf.best_estimator_.get_params()}")

    def evaluate_model(self):
        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val = self.split_data()
        norm_X_train, norm_X_test, norm_X_val = self.feature_normaliser()

        models = [LogisticRegression(), KNeighborsClassifier(),
                  RandomForestClassifier()]

        for model in models:
            model.fit(norm_X_train, self.y_train)
            y_pred_val = model.predict(norm_X_val)

            print(f"Classification metrics for {model}: {classification_report(self.y_val, y_pred_val,
                  target_names=['absense', 'precense'])}")
            # cm = confusion_matrix(self.y_val, y_pred_val)
            # ConfusionMatrixDisplay(cm).plot()

    def voting_class(self):
        vote_clf = VotingClassifier(estimators=[
            ('lr', LogisticRegression()),
            ('knn', KNeighborsClassifier()),
            ('rf', RandomForestClassifier())
        ], voting='hard')

        evaluation = self.evaluate_model(vote_clf)
        return evaluation
