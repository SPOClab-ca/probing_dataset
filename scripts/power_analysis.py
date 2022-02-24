import numpy as np 
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar 
from typing import Tuple
from sklearn.metrics.cluster import contingency_matrix
import json


def compute_prob_table(row1: pd.Series, row2: pd.Series, pred_col_name: str, label_col_name: str) -> Tuple[np.array, int]:
    L1 = np.array(json.loads(row1[pred_col_name])) == np.array(json.loads(row1[label_col_name]))
    L2 = np.array(json.loads(row2[pred_col_name])) == np.array(json.loads(row2[label_col_name]))
    N = len(L1)
    assert len(L1) == len(L2)
    cm = contingency_matrix(L1, L2) / N
    return cm, N


"""
https://github.com/dallascard/NLP-power-analysis/blob/master/notebooks_for_power_calculations/accuracy.ipynb
"""
def power_from_prob_table(prob_table: np.array, dataset_size: int, alpha: float=0.05, r: int=5000) -> Tuple[float, float, float, float]:
    """
    prob_table: 2d np.array. e.g., [[]]
    Run r simulations, each doing a McNemar's test.
    """
    if prob_table[0, 1] == prob_table[1, 0]:
        raise RuntimeError("Power is undefined when the true effect is zero. prob_table is {}".format(str(prob_table)))

    pvals = []
    diffs = []
    for i in range(r):  # number of simulations
        sample = np.random.multinomial(n=dataset_size, pvals=prob_table.reshape((4,))).reshape((2,2))
        acc_diff = (sample[0,1] - sample[1, 0]) / dataset_size
        test_results = mcnemar(sample)
        pvals.append(test_results.pvalue)
        diffs.append(acc_diff)

    true_diff = prob_table[0, 1] - prob_table[1, 0]
    true_sign = np.sign(true_diff) 
    sig_diffs = [d for i, d in enumerate(diffs) if pvals[i] <= alpha]
    power = len([d for i, d in enumerate(diffs) if pvals[i] <= alpha and np.sign(d) == true_sign]) / r
    mean_effect = np.mean(diffs)
    type_m = np.mean(np.abs(sig_diffs) / np.abs(true_diff))
    type_s = np.mean(np.sign(sig_diffs) != true_sign)
    return power, mean_effect, type_m, type_s


def power_from_multiple_rows(df1: pd.DataFrame, df2: pd.DataFrame, pred_col_name: str, label_col_name: str) -> Tuple[float, float, float, float]:
    """
    Consider samples from multiple classification experiments.
    """
    assert len(df1) == len(df2), "Expect two df to have same rows. Got {} and {}".format(len(df1), len(df2))
    df1_ = df1.sort_values(by="seed")
    df2_ = df2.sort_values(by="seed")
    powers = []
    for i in range(len(df1)):
        assert df1_.iloc[i].seed == df2_.iloc[i].seed, "The seeds should be identical!"
        cm, N = compute_prob_table(df1_.iloc[i], df2_.iloc[i], pred_col_name, label_col_name)
        try:
            power, _, _, _ = power_from_prob_table(cm, N, r=int(5000//len(df1)))
        except RuntimeError:
            power = 0
        powers.append(power)
    return np.mean(powers)
