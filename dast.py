"""
DAST (Decision-Aware Segmentation Tree)

Best-first tree growth: starting from a single root leaf, greedily expand
the leaf whose best admissible split gives the largest DR-value gain, until
exactly M leaves are reached or no admissible split exists (min_leaf_size
constraint). Non-positive-gain splits are allowed to reach the target M.
"""

import warnings
import numpy as np
from ground_truth import SegmentEstimate, PopulationSimulator
from utils import estimate_segment_parameters, evaluate_on_validation, compute_node_DR_value


class DASTNode:
    """Node in a DAST tree."""

    def __init__(self, indices, depth=0):
        self.indices  = indices
        self.depth    = depth
        self.value    = None          # V̂_node (DR value), set during build

        # Split info (None for leaves)
        self.split_feature   = None
        self.split_threshold = None
        self.left  = None
        self.right = None
        self.is_leaf = True

        # Segment info (set after build)
        self.segment_id = None


class DASTree:
    """DAST decision tree: best-first growth to M leaves."""

    def __init__(self, x, y, D, gamma, candidate_thresholds,
                 min_leaf_size, algo, use_hybrid_method, action_method: str,
                 action_num=None, include_interactions: bool = False):
        self.x               = x
        self.y               = y
        self.D               = D
        self.gamma           = gamma
        self.H               = candidate_thresholds
        self.min_leaf_size   = min_leaf_size
        self.algo            = algo
        self.use_hybrid_method = use_hybrid_method
        self.action_num      = action_num
        self.action_method   = action_method
        self.include_interactions = include_interactions

        self.root       = None
        self.leaf_nodes = []   # maintained in sync throughout build

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(self, M, debug=False):
        """
        Best-first grow tree to exactly M leaves.
        Implements Algorithm 1 (BuildTree) from the paper.

        Parameters
        ----------
        M     : target number of leaves
        debug : print split details when True
        """
        all_indices = np.arange(self.x.shape[0])
        self.root = DASTNode(all_indices, depth=0)
        self.leaf_nodes = [self.root]

        if debug:
            print(f"\n{'='*60}")
            print(f"Building DAST Tree  (N={len(all_indices)}, target M={M})")
            print(f"{'='*60}")

        while len(self.leaf_nodes) < M:
            best_global_gain = -np.inf
            best_leaf        = None
            best_split       = None   # (feature, threshold, left_idx, right_idx)

            # Snapshot to avoid iterating over a list that is modified after the loop
            for leaf in list(self.leaf_nodes):
                split, gain, V_node = self._find_best_split(leaf)
                leaf.value = V_node   # store V̂_node (cached; reused if leaf survives)

                if gain > best_global_gain:
                    best_global_gain = gain
                    best_leaf  = leaf
                    best_split = split

            # Hard stop: no admissible split exists anywhere in the tree
            if best_leaf is None or best_split is None:
                warnings.warn(
                    f"DAST terminates early with {len(self.leaf_nodes)} leaves "
                    f"(target M={M}): no admissible split satisfies min_leaf_size constraints.",
                    stacklevel=2,
                )
                break

            # Soft warning: best available gain is non-positive, but we keep splitting
            if best_global_gain <= 0:
                warnings.warn(
                    f"DAST: non-positive gain split (gain={best_global_gain:.4f}) at "
                    f"{len(self.leaf_nodes)} leaves (target M={M}).",
                    stacklevel=2,
                )

            # Apply the best split
            feat, thresh, left_idx, right_idx = best_split
            left_node  = DASTNode(left_idx,  depth=best_leaf.depth + 1)
            right_node = DASTNode(right_idx, depth=best_leaf.depth + 1)

            best_leaf.is_leaf          = False
            best_leaf.split_feature    = feat
            best_leaf.split_threshold  = thresh
            best_leaf.left             = left_node
            best_leaf.right            = right_node

            self.leaf_nodes.remove(best_leaf)
            self.leaf_nodes.append(left_node)
            self.leaf_nodes.append(right_node)

            if debug:
                print(f"  Split leaf depth={best_leaf.depth} (N={len(best_leaf.indices)})"
                      f"  X[{feat}] <= {thresh:.4f}"
                      f"  → L={len(left_idx)}, R={len(right_idx)}"
                      f"  gain={best_global_gain:.4f}"
                      f"  leaves={len(self.leaf_nodes)}")

        if debug:
            print(f"\nTree built: {len(self.leaf_nodes)} leaves")

    def fit_leaves(self, data, customers, debug=False):
        """
        After build(), assign segment IDs, estimate parameters,
        and link training customers to their segment.

        Returns list of SegmentEstimate objects.
        """
        # Ensure every leaf has a stored V̂_node.
        # After the build loop, the final two newly-created leaves always have value=None
        # (they were added AFTER the last iteration's inner for-loop ran).
        # M=1 (root never entered the loop) is also handled here.
        for leaf in self.leaf_nodes:
            if leaf.value is None:
                leaf.value = compute_node_DR_value(
                    self.y, self.D, self.gamma, leaf.indices,
                    self.use_hybrid_method,
                    X=self.x,
                    action_method=self.action_method,
                    action_num=self.action_num,
                    include_interactions=self.include_interactions,
                )

        for i, node in enumerate(self.leaf_nodes):
            node.segment_id = i

        segments = [
            self._fit_segment_and_assign(
                customers, node.indices, data, node.segment_id
            )
            for node in self.leaf_nodes
        ]

        if debug:
            print("\nSegment details:")
            for seg, node in zip(segments, self.leaf_nodes):
                print(f"  Seg {seg.segment_id}: N={len(node.indices)}"
                      f"  tau={seg.est_tau:+.4f}  action={seg.est_action}"
                      f"  depth={node.depth}")

        return segments

    def predict_segment(self, customers, segment_dict):
        """Assign each customer to a segment by traversing the tree."""
        for cust in customers:
            node = self.root
            while not node.is_leaf:
                if cust.x[node.split_feature] <= node.split_threshold:
                    node = node.left
                else:
                    node = node.right
            cust.est_segment[self.algo] = segment_dict[node.segment_id]

    # ------------------------------------------------------------------
    # Core: FindBestSplit  (Algorithm 1, FindBestSplit sub-routine)
    # ------------------------------------------------------------------

    def _find_best_split(self, node):
        """
        Find the best admissible split for a leaf node.

        Returns
        -------
        split  : (feature, threshold, left_idx, right_idx) or None
        gain   : best gain achieved (-inf if no admissible split found)
        V_node : DR value of the node itself
        """
        indices = node.indices
        # Reuse cached value if available (avoids redundant computation across iterations)
        if node.value is not None:
            V_node = node.value
        else:
            V_node = compute_node_DR_value(
                self.y, self.D, self.gamma, indices,
                self.use_hybrid_method,
                X=self.x,
                action_method=self.action_method,
                action_num=self.action_num,
                include_interactions=self.include_interactions,
            )

        best_gain          = -np.inf
        best_j, best_t     = None, None
        best_left          = None
        best_right         = None
        best_var_reduction = -np.inf   # used as tie-breaker

        for j in range(self.x.shape[1]):
            for t in self.H[j]:
                left_idx  = indices[self.x[indices, j] <= t]
                right_idx = indices[self.x[indices, j] >  t]

                if not (self._check_leaf_constraints(left_idx) and
                        self._check_leaf_constraints(right_idx)):
                    continue

                V_left  = compute_node_DR_value(
                    self.y, self.D, self.gamma, left_idx,
                    self.use_hybrid_method,
                    X=self.x,
                    action_method=self.action_method,
                    action_num=self.action_num,
                    include_interactions=self.include_interactions,
                )
                V_right = compute_node_DR_value(
                    self.y, self.D, self.gamma, right_idx,
                    self.use_hybrid_method,
                    X=self.x,
                    action_method=self.action_method,
                    action_num=self.action_num,
                    include_interactions=self.include_interactions,
                )
                gain = V_left + V_right - V_node

                if gain > best_gain + 1e-9:
                    # Strictly better gain
                    best_gain = gain
                    best_j, best_t         = j, t
                    best_left, best_right  = left_idx, right_idx
                    best_var_reduction     = self._compute_variance_reduction(
                        indices, left_idx, right_idx)

                elif abs(gain - best_gain) <= 1e-9:
                    # Tied gain: prefer larger covariate variance reduction
                    var_red = self._compute_variance_reduction(indices, left_idx, right_idx)
                    if var_red > best_var_reduction:
                        best_j, best_t         = j, t
                        best_left, best_right  = left_idx, right_idx
                        best_var_reduction     = var_red

        if best_j is None:
            return None, -np.inf, V_node

        return (best_j, best_t, best_left, best_right), best_gain, V_node

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_leaf_constraints(self, indices):
        """Every action 0..action_num-1 must appear >= min_leaf_size times."""
        D_sub = self.D[indices]
        for a in range(self.action_num):
            if np.sum(D_sub == a) < self.min_leaf_size:
                return False
        return True

    def _compute_variance_reduction(self, parent_idx, left_idx, right_idx):
        parent_var = self._compute_covariate_variance(parent_idx)
        left_var   = self._compute_covariate_variance(left_idx)
        right_var  = self._compute_covariate_variance(right_idx)
        n = len(parent_idx)
        if n == 0:
            return 0.0
        weighted = (len(left_idx) * left_var + len(right_idx) * right_var) / n
        return parent_var - weighted

    def _compute_covariate_variance(self, indices):
        if len(indices) < 2:
            return 0.0
        return np.var(self.x[indices], axis=0, ddof=1).sum()

    def _fit_segment_and_assign(self, customers, indices, data, segment_id):
        """Estimate parameters for one leaf and assign to its training customers."""
        est_tau, est_action = estimate_segment_parameters(
            data['X'], data['D'], data['Y'],
            method=self.action_method,
            gamma=self.gamma,
            indices=indices,
            action_num=self.action_num,
            include_interactions=self.include_interactions,
        )
        segment = SegmentEstimate(est_tau, est_action, segment_id)

        for i in indices:
            customers[i].est_segment[self.algo] = segment

        return segment

    def _get_leaf_nodes(self):
        """Return current leaf list (for compatibility)."""
        return list(self.leaf_nodes)


# ==============================================================
# Main Interface Function
# ==============================================================

def DAST_segment_and_estimate(pop: PopulationSimulator, n_segments,
                               min_leaf_size, algo,
                               use_hybrid_method, action_method: str,
                               include_interactions: bool = False,
                               debug=False):
    """
    Main interface for DAST algorithm.

    Parameters
    ----------
    pop               : PopulationSimulator with train/val customers and gamma
    n_segments        : target number of leaves M
    min_leaf_size     : minimum samples per action in each leaf
    algo              : algorithm key string (e.g. 'dast')

    use_hybrid_method : DR value computation mode
    debug             : verbose output

    Returns
    -------
    tree         : fitted DASTree
    val_score    : validation DR score (None if no val customers)
    segment_dict : {segment_id: SegmentEstimate}
    """
    data_train = {
        "X":     np.array([cust.x   for cust in pop.train_customers]),
        "D":     np.array([cust.D_i for cust in pop.train_customers]).reshape(-1, 1),
        "Y":     np.array([cust.y   for cust in pop.train_customers]).reshape(-1, 1),
        "Gamma": pop.gamma_train,
    }

    # Candidate thresholds: quantile-based binning per feature
    B = 200
    H = {}
    for j in range(data_train["X"].shape[1]):
        sv       = np.sort(np.unique(data_train["X"][:, j]))
        n_unique = len(sv)
        if n_unique <= 1:
            H[j] = np.array([])   # no valid threshold when feature is constant
        elif n_unique <= B:
            H[j] = (sv[:-1] + sv[1:]) / 2.0
        else:
            q_idx = [int(np.floor(k / B * n_unique)) for k in range(1, B)]
            q_idx = sorted(set(min(i, n_unique - 1) for i in q_idx))
            qv    = sv[q_idx]
            H[j]  = (qv[:-1] + qv[1:]) / 2.0

    tree = DASTree(
        x=data_train["X"],
        y=data_train["Y"],
        D=data_train["D"],
        gamma=data_train["Gamma"],
        candidate_thresholds=H,
        min_leaf_size=min_leaf_size,
        algo=algo,
        use_hybrid_method=use_hybrid_method,
        action_num=pop.action_num,
        action_method=action_method,
        include_interactions=include_interactions,
    )

    tree.build(M=n_segments, debug=debug)
    segments = tree.fit_leaves(data_train, pop.train_customers, debug=debug)

    pop.est_segments_list[algo] = segments
    segment_dict = {seg.segment_id: seg for seg in segments}

    val_score = None
    if len(pop.val_customers) > 0:
        tree.predict_segment(pop.val_customers, segment_dict)
        val_score = evaluate_on_validation(pop, algo=algo, Gamma_val=pop.gamma_val)
        if debug:
            print(f"Validation score: {val_score:.4f}")

    return tree, val_score, segment_dict
