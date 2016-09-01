from sklearn.metrics import fbeta_score as metric
from sklearn.metrics import make_scorer


def score(y_true, y_pred, **kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight=None
    try:
        beta = kwargs["1"]
    except:
        beta=1.0
    try:
        labels = kwargs["2"]
    except:
        labels=None
    try:
        pos_label = kwargs["3"]
    except:
        pos_label = 1
    try:
        average = kwargs["4"]
    except:
        average = "binary"
    score = metric(y_true, y_pred, beta, sample_weight=sample_weight, labels=labels, pos_label=pos_label, average=average)
    return score


def get_scorer(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight=None
    try:
        beta = kwargs["1"]
    except:
        beta=1.0
    try:
        labels = kwargs["2"]
    except:
        labels=None
    try:
        pos_label = kwargs["3"]
    except:
        pos_label = 1
    try:
        average = kwargs["4"]
    except:
        average = "binary"
    return make_scorer(metric, greater_is_better=True, beta=beta, sample_weight=sample_weight, labels=labels,
                       pos_label=pos_label, average=average)