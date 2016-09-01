from sklearn.metrics import jaccard_similarity_score as metric
from sklearn.metrics import make_scorer


def score(y_true, y_pred, **kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight = None
    score = metric(y_true, y_pred, sample_weight=sample_weight)
    return score


def get_scorer(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight = None
    return make_scorer(metric, greater_is_better=True, sample_weight=sample_weight)