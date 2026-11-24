from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import entropy
from sklearn.metrics.cluster import contingency_matrix
import numpy as np


def variation_of_information(true_labels, pred_labels):
    """
    Compute Variation of Information (VI) between two clusterings.
    """
    contingency = contingency_matrix(true_labels, pred_labels)
    joint = contingency / np.sum(contingency)
    pi = np.sum(joint, axis=1)
    pj = np.sum(joint, axis=0)

    H_true = entropy(pi)
    H_pred = entropy(pj)
    I = np.sum(joint * np.log((joint + 1e-10) / (pi[:, None] * pj[None, :] + 1e-10)))

    VI = H_true + H_pred - 2 * I
    return VI


def structure_oracle(true_labels, pred_labels):
    """
    Compute structure recovery metrics comparing true vs estimated clusters.

    Returns:
    - dict with metrics: ARI, NMI
    """
    true_labels = np.array(true_labels).reshape(-1)
    pred_labels = np.array(pred_labels).reshape(-1)

    return {
        'ARI': adjusted_rand_score(true_labels, pred_labels),
        'NMI': normalized_mutual_info_score(true_labels, pred_labels),
    }


def oracle_profit_on_customers(customers):
    """
    Compute the true per-customer oracle (first-best) total profit.

    For each customer x_i the oracle picks the action that maximises
    cust.expected_outcome(a) over ALL actions a.  Because outcome depends
    on both tau[a] AND delta[a]@x_i, using argmax(tau) alone is wrong
    when interaction effects (delta) are present.

    This quantity is guaranteed to be >= any algorithm's profit on the
    same customer set, making it a valid normalising denominator.

    Parameters
    ----------
    customers  : list of Customer_implement (or Customer_pilot)

    Returns
    -------
    float  total oracle expected profit
    """
    total = 0.0
    for cust in customers:
        action_num = len(cust.true_segment.tau)
        total += max(cust.expected_outcome(a) for a in range(action_num))
    return total


def policy_oracle(customers, algo):
    """
    Compute policy evaluation metrics:
    - Manager profit
    - Oracle profit
    - Regret
    - Mis-treatment rate

    Parameters:
    - customers: list of Customer objects (each linked to true and estimated segments)
    - algo: algorithm name key into cust.est_segment

    Returns:
    - dict with metrics: manager_profit, oracle_profit, regret, mistreatment_rate
    """
    manager_profit = 0.0
    oracle_profit  = 0.0
    mistreated     = 0

    for cust in customers:
        est_seg    = cust.est_segment[algo]
        action_num = len(cust.true_segment.tau)

        # True per-customer oracle: best action for this customer's x
        oracle_profit_i = max(cust.expected_outcome(a) for a in range(action_num))
        oracle_action   = max(range(action_num), key=cust.expected_outcome)

        # Manager policy
        if est_seg.est_action == 404:
            manager_profit_i = oracle_profit_i
        else:
            manager_profit_i = cust.expected_outcome(est_seg.est_action)

        # Mistreatment if manager's action ≠ oracle
        if est_seg.est_action != oracle_action and est_seg.est_action != 404:
            mistreated += 1

        oracle_profit  += oracle_profit_i
        manager_profit += manager_profit_i

    return {
        "manager_profit":    manager_profit,
        "oracle_profit":     oracle_profit,
        "regret":            (oracle_profit - manager_profit) / len(customers),
        "mistreatment_rate": mistreated / len(customers),
    }
