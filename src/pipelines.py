from sklearn.base import BaseEstimator, TransformerMixin


class rare_category_grouper(BaseEstimator, TransformerMixin):
    def __init__(self, cols, top_n=10, other_label="others"):
        self.cols = cols
        self.top_n = top_n
        self.other_label = other_label
        self.top_categories_ = {}

    def fit(self, X, y=None):
        X = X.copy()

        self.top_categories_ = {}
        for col in self.cols:
            if col in X.columns:
                self.top_categories_[col] = (
                    X[col].value_counts().nlargest(self.top_n).index
                )

        return self

    def transform(self, X):
        X = X.copy()

        for col, top_vals in self.top_categories_.items():
            if col in X.columns:
                X[col] = X[col].apply(
                    lambda x: x if x in top_vals else self.other_label
                )

        return X


