""""
AutoMap class
"""

# import
import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


class AutoMap:
    """
    :params:
    train_files: list of files to load for training the prediction pipeline on
    predict_files: list of files to read and predict the parameters for
    output_files: list of files to save the predictions to, where length equals length of predict_files
    pipe_file: Previously created compatible pickled tuple of fitted (Pipeline, LabelEncoder) objects


    :args:



    """

    def __init__(self,
                 train_files: list = None,
                 predict_files: list = None,
                 output_files: list = None,
                 pipe_file: str = None):
        self.__version__ = '0.2.2'
        self.pipe_version = self.__version__
        self.datetime_created = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')  # hidden in Pipe creation

        if pipe_file:
            try:
                self.pipe, self.le, self.pipe_version, self.datetime_created = joblib.load(pipe_file)
                print(f'Loaded pipeline: {self.pipe}')
            except OSError:
                raise
        else:
            try:
                hard_coded_pipe = '../data/pipes/pipeline.pipe'
                if os.path.isfile(hard_coded_pipe):
                    self.pipe, self.le, self.pipe_version, self.datetime_created = joblib.load(hard_coded_pipe)
                else:
                    self.pipe = None
                    self.le = LabelEncoder()
            except:
                self.pipe = None
                self.le = LabelEncoder()

        if train_files is None:
            hard_coded_file = '../data/input/combined.csv'
            if os.path.isfile(hard_coded_file):
                self.df = pd.read_csv(hard_coded_file)

        # column names to use in training/predicting
        self.source = 'parameter_name'
        self.target = 'pacmed_subname'
        self.pred = 'predicted_subname'

        # if no label is given, impute with index[0] (unmapped)
        self.unlabeled = ['unmapped', 'microbiology']  # used for validation to filter out unlabeled

        # initial r'\w+' but 5% performance gain when underscores are omitted
        self.preprocess_text_regex_expression = r'[a-zA-Z0-9]+'
        return

    def preprocess_text(self, text):
        # Tokenise words while ignoring punctuation
        tokeniser = RegexpTokenizer(self.preprocess_text_regex_expression)
        tokens = tokeniser.tokenize(text)

        # Lowercase and lemmatise
        lemmatiser = WordNetLemmatizer()
        lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]

        # Remove stop words
        # keywords= [lemma for lemma in lemmas if lemma not in stopwords.words('english')]
        # return keywords
        return lemmas

    def create_pipe(self,
                    X=pd.Series(),
                    y=pd.Series(),
                    estimator=SGDClassifier(random_state=123),
                    grid: dict = None,
                    cv: int = 10,
                    n_jobs: int = None,
                    save: bool = False,
                    prefix=None):
        """
        Create the pipe object used to train and test text data
        """

        if y.isna().sum() > 0:
            y = y.fillna(self.unlabeled[0])

        # ensure labels are encoded
        self.le = LabelEncoder()
        self.le.fit(y=y.unique())

        # Create an instance of TfidfVectorizer
        vectoriser = TfidfVectorizer(analyzer=self.preprocess_text)

        # Fit to the data and transform to feature matrix
        X_train_tfidf = vectoriser.fit_transform(X)

        # try an initial accuracy before hyperparameter optimization
        clf = estimator
        # clf = SGDClassifier(random_state=123)
        # clf_scores = cross_val_score(clf, X_train_tfidf, self.y_train, cv=10)
        # print(clf_scores)
        # print("SGDClassfier Accuracy: %0.2f (+/- %0.2f)" % (clf_scores.mean(), clf_scores.std() * 2))

        if grid is None:
            grid = {'fit_intercept': [True, False],
                    'early_stopping': [True, False],
                    'loss': ['log', 'modified_huber', 'perceptron', 'huber', 'squared_loss', 'epsilon_insensitive',
                             'squared_epsilon_insensitive'],
                    # ['hinge', 'log', 'squared_hinge'], #PM squared_loss --> squared_error in v1.2
                    'penalty': ['l2', 'l1', 'none']}

            # Reduce to optimal grid for rerunning code
            grid = {'fit_intercept': [True],
                    'early_stopping': [False],
                    'loss': ['modified_huber'],
                    'penalty': ['l2']}

        # retry the SGDClassifier training with param_grid
        search = GridSearchCV(estimator=clf, param_grid=grid, cv=cv, n_jobs=n_jobs)
        search.fit(X_train_tfidf, y)

        # grid_sgd_clf_scores = cross_val_score(search.best_estimator_, X_train_tfidf, self.y_train, cv=5)
        # print(grid_sgd_clf_scores)
        # print("SGDClassifier optimal grid Accuracy: %0.2f (+/- %0.2f)" % (
        # grid_sgd_clf_scores.mean(), grid_sgd_clf_scores.std() * 2))

        # create Pipeline with vectoriser and optimal classifier
        self.pipe = Pipeline([('vectoriser', vectoriser),
                              ('classifier', search)])  # clf

        # fit the pipeline to the full training data
        self.pipe.fit(X, self.le.transform(y.values))

        # save pipe to file to prevent rerunning the same pipelines
        if prefix is None:
            prefix = ''
        if save:
            f_name = f'./data/pipes/{prefix}__{datetime.now().strftime("%Y%m%d%H%M%S")}.pipe'
            joblib.dump((self.pipe,
                         self.le,
                         self.pipe_version,
                         self.datetime_created,
                         ),
                        f_name,
                        compress=('gzip', 3),
                        protocol=5)
            print(f"Pipeline saved to: {f_name}")

        return self.pipe

    def save_pipe(self, f_name):
        joblib.dump((self.pipe, self.le), f_name)
        print(f"Pipeline saved to: {f_name}")

    def load_pipe(self, f_name):
        if os.path.isfile(f_name):
            self.pipe, self.le = joblib.load(f_name)
        else:
            self.pipe = None
            self.le = LabelEncoder()
        print(f"Pipeline loaded from: {f_name}")

    def predict_proba_transformed(self, X, **predict_proba_params):

        if isinstance(X, pd.Series):
            probs = self.pipe.predict_proba(X, **predict_proba_params)
            id_vars = [X.name]
            X = pd.DataFrame(X)
        else:
            probs = self.pipe.predict_proba(X[X.columns[1]], **predict_proba_params)
            id_vars = list(X.columns)

        df_probs = X.copy()
        df_probs[self.le.classes_] = probs
        # df_probs = pd.concat([X, pd.DataFrame(data=probs, columns=self.le.classes_)], axis=1, ignore_index=True)

        df_probs_melt = pd.melt(df_probs, id_vars=id_vars, value_vars=self.le.classes_, var_name='label')
        return df_probs_melt.loc[df_probs_melt['value'] > 0].sort_values(id_vars + ['value']).copy()


if __name__ == '__main__':
    am = AutoMap()
    df_full = pd.read_csv('../data/input/archive/combined_mappings_covid.csv')
    df = df_full.head(1000)
    am.create_pipe(X=df['parameter_name'], y=df['pacmed_subname'])
    a = am.predict_proba_transformed(X=df['parameter_name'])
    print('done')