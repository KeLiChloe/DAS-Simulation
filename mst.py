# mst_tree.py

import numpy as np
from ground_truth import SegmentEstimate, PopulationSimulator
from utils import estimate_segment_parameters, evaluate_on_validation, build_design_matrix
from sklearn.linear_model import LogisticRegression

import random


def _node_residual(Z_m, y_m, D_m, action_num):
    """
    Compute Bernoulli NLL impurity from pre-sliced arrays.
    Called O(d × B) times per tree node.
    """
    # Strict check: every action must be present
    if action_num is not None:
        for a in range(action_num):
            if not np.any(D_m == a):
                return np.inf

    y_int = y_m.astype(int)
    if len(np.unique(y_int)) == 1:
        return 0.0
    model = LogisticRegression(
        fit_intercept=False, solver='lbfgs',
        max_iter=100, tol=1e-3, C=1e6,
    )
    try:
        model.fit(Z_m, y_int)
        proba   = model.predict_proba(Z_m)
        p_true  = np.clip(proba[np.arange(len(y_int)), y_int], 1e-9, 1 - 1e-9)
        return  -np.sum(np.log(p_true))
    except Exception:
        p_mean = np.clip(np.mean(y_int), 1e-9, 1 - 1e-9)
        return -np.sum(y_int * np.log(p_mean) + (1 - y_int) * np.log(1 - p_mean))

class MSTNode:
    def __init__(self, indices, depth=0):
        self.indices = indices            # Indices of customers in this node
        self.depth = depth
        self.split_feature = None
        self.split_threshold = None
        self.left = None
        self.right = None
        self.is_leaf = True
        self.segment_id = None            # Set after tree construction
        self.tau_hat = None
        self.value = None                
    
    def prune(self):
        self.left = None
        self.right = None
        self.is_leaf = True

class MSTree:
    def __init__(self, x, y, D, Z, candidate_thresholds, min_leaf_size, epsilon, max_depth, algo, action_num, action_method: str, gamma=None, include_interactions: bool = False):
        self.x  = x
        self.y  = y.ravel()          # keep as 1-D for fast indexing
        self.D  = D.ravel().astype(int)
        self.Z  = Z                  # precomputed design matrix (N, p)
        self.H  = candidate_thresholds
        self.min_leaf_size = min_leaf_size
        self.epsilon       = epsilon
        self.max_depth     = max_depth

        self.root       = None
        self.leaf_nodes = []

        self.algo       = algo
        self.action_num = action_num
        self.action_method = action_method
        self.gamma = gamma
        self.include_interactions = include_interactions

    def _eval(self, indices):
        """Evaluate impurity for a node using pre-sliced Z / y / D."""
        return _node_residual(
            self.Z[indices], self.y[indices], self.D[indices],
            self.action_num,
        )

    def build(self, include_interactions):
        N = self.x.shape[0]
        all_indices = np.arange(N)
        self.root = self._grow_node(all_indices, depth=0, init_value=None)
        
    
    def prune_to_segments_and_estimate(self, M, data, customers, include_interactions):
        """
        Prune the tree to have exactly M leaf nodes.
        Then fit OLS and assign SegmentEstimate to customers.
        """
        current_leaves = self._get_leaf_nodes()

        # if len(current_leaves) < M:
        #     raise ValueError(f"Current leaves ({len(current_leaves)}) are already less than or equal to M ({M}). No pruning needed.")
        
        while len(current_leaves) > M:
            max_gain = -np.inf
            candidate_node = None
            for node in self._get_internal_nodes_with_leaf_children():
                gain = self._compute_pruning_gain(node)
                if gain > max_gain:
                    max_gain = gain
                    candidate_node = node
            if candidate_node:
                candidate_node.prune()
            current_leaves = self._get_leaf_nodes()

        self.leaf_nodes = current_leaves
        for i, node in enumerate(self.leaf_nodes):
            node.segment_id = i

        final_segments = []
        for node in self.leaf_nodes:
            segment = self._fit_segment_and_assign(customers, node.indices, data, node.segment_id, include_interactions)
            final_segments.append(segment)
        return final_segments

    def predict_segment(self, customers, segment_dict):
        """
        Predicts and assigns segment IDs for a list of new Customer objects based on their covariates.

        Args:
            customers: list of Customer objects
        """
        for cust in customers:
            xi = cust.x
            node = self.root
            while not node.is_leaf:
                if xi[node.split_feature] <= node.split_threshold:
                    node = node.left
                else:
                    node = node.right

            # Assign SegmentEstimate or segment_id (if lookup not needed)
            segment_obj = segment_dict[node.segment_id]
            cust.est_segment[f"{self.algo}"] = segment_obj


    def _grow_node(self, indices, depth, init_value=None):
        node = MSTNode(indices, depth)
        # Reuse value computed by the parent during split search (avoid double work)
        node.value = init_value if init_value is not None else self._eval(indices)

        if depth == self.max_depth:
            self.leaf_nodes.append(node)
            return node

        best_gain   = -np.inf
        best_split  = None        # (j, t, left_idx, right_idx, lv, rv)
        valid_splits = []

        x_node = self.x[indices]  # local view – avoids repeated fancy indexing

        for j in range(self.x.shape[1]):
            xj = x_node[:, j]
            for t in self.H[j]:
                mask_l = xj <= t
                mask_r = ~mask_l
                left_idx  = indices[mask_l]
                right_idx = indices[mask_r]

                if not (self._check_leaf_constraints(left_idx) and
                        self._check_leaf_constraints(right_idx)):
                    continue

                lv   = self._eval(left_idx)
                rv   = self._eval(right_idx)
                gain = node.value - (lv + rv)
                valid_splits.append((gain, j, t, left_idx, right_idx, lv, rv))

                if gain >= best_gain:
                    best_gain  = gain
                    best_split = (j, t, left_idx, right_idx, lv, rv)

        if best_split is None:
            if valid_splits:
                _, j_r, t_r, left_idx, right_idx, lv, rv = random.choice(valid_splits)
                node.is_leaf = False
                node.split_feature   = j_r
                node.split_threshold = t_r
                node.left  = self._grow_node(left_idx,  depth + 1, init_value=lv)
                node.right = self._grow_node(right_idx, depth + 1, init_value=rv)
            else:
                self.leaf_nodes.append(node)
            return node

        j, t, left_idx, right_idx, lv, rv = best_split
        node.is_leaf = False
        node.split_feature   = j
        node.split_threshold = t
        node.left  = self._grow_node(left_idx,  depth + 1, init_value=lv)
        node.right = self._grow_node(right_idx, depth + 1, init_value=rv)
        return node

    
    def _check_leaf_constraints(self, indices):
        """Check if leaf has minimum required samples for ALL possible actions."""
        D_sub = self.D[indices]
        
        # STRICT: All actions 0..action_num-1 must each have at least min_leaf_size samples
        for a in range(self.action_num):
            if np.sum(D_sub == a) < self.min_leaf_size:
                return False
        
        return True

    
    def _get_leaf_nodes(self):
        return self._gather_nodes(self.root, is_leaf=True)

    def _get_internal_nodes_with_leaf_children(self):
        def condition(node):
            return (
                not node.is_leaf and 
                node.left and node.right and 
                node.left.is_leaf and node.right.is_leaf
            )
        return self._gather_nodes(self.root, condition=condition)

    def _gather_nodes(self, node, is_leaf=False, condition=None):
        if node is None:
            return []
        if condition:
            return (
                [node] if condition(node) else []
            ) + self._gather_nodes(node.left, condition=condition) + self._gather_nodes(node.right, condition=condition)
        if is_leaf and node.is_leaf:
            return [node]
        if not is_leaf and not node.is_leaf:
            return [node]
        return self._gather_nodes(node.left, is_leaf=is_leaf) + self._gather_nodes(node.right, is_leaf=is_leaf)

    def _compute_pruning_gain(self, node):
        """
        Compute the gain of pruning (i.e., merging left and right back to parent).
        
        Pruning gain = parent_value - (left_value + right_value)
        
        For MST, value = residual (lower is better).
        If gain > 0, pruning increases residual (we lose fit quality).
        If gain < 0, keeping the split is better (reduces residual).
        
        We want to prune nodes with the highest gain (least valuable splits).
        """
        left, right = node.left, node.right
        if left is None or right is None:
            return np.inf
        
        # Use cached values instead of recomputing
        # All node values were computed during tree building
        gain = node.value - (left.value + right.value)
        
        return gain

    def _fit_segment_and_assign(self, customers, indices, data, segment_id, include_interactions):
        est_tau, est_action = estimate_segment_parameters(
            data['X'], data['D'], data['Y'],
            method=self.action_method,
            gamma=self.gamma,
            indices=indices,
            action_num=self.action_num,
            include_interactions=include_interactions,
        )
        segment = SegmentEstimate(est_tau, est_action, segment_id)
        for i in indices:
            customers[i].est_segment[f"{self.algo}"] = segment
        return segment



def MST_segment_and_estimate(pop: PopulationSimulator, n_segments, max_depth, min_leaf_size, epsilon, algo, include_interactions, action_method: str, threshold_grid=None):
    # Prepare training data
    X = np.array([cust.x   for cust in pop.train_customers])
    D = np.array([cust.D_i for cust in pop.train_customers])
    Y = np.array([cust.y   for cust in pop.train_customers])
    data_train = {"X": X, "D": D.reshape(-1, 1), "Y": Y.reshape(-1, 1)}

    # Precompute design matrix ONCE — reused for every candidate split
    # Multi-arm D is one-hot (action 0 reference), not a numeric scalar.
    Z = build_design_matrix(X, D, include_interactions, action_num=pop.action_num)

    # Generate candidate thresholds using quantile-based binning (B bins)
    B = 50
    H = {}
    for j in range(X.shape[1]):
        sorted_values = np.sort(np.unique(X[:, j]))
        N_unique = len(sorted_values)
        if N_unique <= B:
            H[j] = sorted_values
        else:
            quantile_indices = sorted(set(
                min(int(np.floor(k / B * N_unique)), N_unique - 1)
                for k in range(1, B)
            ))
            H[j] = sorted_values[quantile_indices]

    # Build MST (Z is passed in; no design matrix is rebuilt inside the tree)
    tree = MSTree(
        x=X, y=Y, D=D, Z=Z,
        candidate_thresholds=H,
        min_leaf_size=min_leaf_size,
        epsilon=epsilon,
        max_depth=max_depth,
        algo=algo,
        action_num=pop.action_num,
        action_method=action_method,
        gamma=pop.gamma_train,
        include_interactions=include_interactions,
    )
    tree.build(include_interactions)
    
    # Prune to M leaves and estimate parameters
    pruned_segments = tree.prune_to_segments_and_estimate(n_segments, data_train, pop.train_customers, include_interactions)
    pop.est_segments_list[f"{algo}"] = pruned_segments
    segment_dict = {seg.segment_id: seg for seg in pruned_segments}
    
    
    # Assign each validate customer to estimated segment
    val_score = None
    if len(pop.val_customers) > 0:
        tree.predict_segment(pop.val_customers, segment_dict)
        Gamma_val = pop.gamma_val
        val_score = evaluate_on_validation(pop, algo=algo, Gamma_val=Gamma_val)
        
    return tree, val_score, segment_dict


