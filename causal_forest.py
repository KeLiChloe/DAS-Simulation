"""
Causal Forest Implementation using econml package.

Causal Forest is an individual-level treatment effect estimation method,
not a segmentation-based approach. It estimates heterogeneous treatment effects
(HTE) for each individual and recommends the optimal action.

Workflow is similar to meta-learners (T-learner, S-learner, X-learner):
1. Fit causal forest on training data
2. Predict individual treatment effects on implementation customers
3. Recommend action based on predicted treatment effect sign

Using econml (Python) instead of R's grf for:
- Faster execution (no R interface overhead)
- Pure Python implementation
- Better integration with scikit-learn ecosystem
"""

import numpy as np
from econml.dml import CausalForestDML


def causal_forest_predict(implement_customers, x_mat, D_vec, y_vec, 
                          n_estimators=8, n_estimators_nuisance=8, 
                          min_samples_leaf=10, max_depth=3):
    """
    Causal Forest implementation using econml's CausalForestDML.
    
    Estimates individual-level heterogeneous treatment effects (HTE) and recommends
    optimal action for each customer.
    
    For binary treatment (D ∈ {0,1}):
    - Fits a causal forest: τ̂(x) = E[Y(1) - Y(0) | X=x]
    - Recommends treatment (action=1) if τ̂(x) ≥ 0, else control (action=0)
    
    For multi-arm treatment (D ∈ {0,1,2,...,k-1}):
    - Uses action 0 as baseline
    - For each action a, estimates τ̂_a(x) = E[Y(a) - Y(0) | X=x]
    - Recommends action with highest predicted outcome
    
    Parameters:
    implement_customers : list
        List of implementation customers
    x_mat : array-like, shape (n_samples, n_features)
        Feature matrix from training data
    D_vec : array-like, shape (n_samples,)
        Action assignment vector from training data
    y_vec : array-like, shape (n_samples,)
        Outcome vector from training data
    n_estimators : int, default=8
        Number of trees in the causal forest (main effect estimation)
    n_estimators_nuisance : int, default=8
        Number of trees in nuisance models (outcome and propensity models)
    min_samples_leaf : int, default=5
        Minimum number of samples in leaf nodes
    max_depth : int or None, default=None
        Maximum depth of trees (None means unlimited)
        
    Returns:
    seg_labels_impl : array-like, shape (n_impl_samples,)
        Recommended action for each implementation customer
    action_identity : array-like, shape (n_actions,)
        Action identity mapping: action_identity[i] = i
    """
    from sklearn.ensemble import RandomForestRegressor
    
    # Identify unique actions
    unique_actions = np.unique(D_vec)
    n_actions = len(unique_actions)
    
    # Get implementation customer features
    X_impl = np.array([cust.x for cust in implement_customers])
    n_impl = X_impl.shape[0]
    
    if n_actions == 2:
        # ========== Binary treatment case ==========
        
        # Fit causal forest using econml
        # CausalForestDML uses double machine learning to debias the estimates
        cforest = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=n_estimators_nuisance, 
                min_samples_leaf=min_samples_leaf, 
                random_state=42
            ),
            model_t=RandomForestRegressor(
                n_estimators=n_estimators_nuisance, 
                min_samples_leaf=min_samples_leaf, 
                random_state=42
            ),
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=42,
            verbose=0
        )
        
        # Fit on training data
        cforest.fit(Y=y_vec, T=D_vec, X=x_mat)
        
        # Predict treatment effects: τ̂(x) = E[Y(1) - Y(0) | X=x]
        tau_hat = cforest.effect(X_impl).flatten()
        
        # Recommend action: treatment if τ̂(x) ≥ 0, control otherwise
        seg_labels_impl = (tau_hat >= 0).astype(int)
        action_identity = unique_actions.astype(int)

    else:
        
        # Check if action 0 exists (needed as baseline)
        if 0 not in unique_actions:
            raise ValueError("Causal forest requires action 0 as baseline for multi-arm treatment")
        
        # For each non-baseline action, estimate treatment effect vs baseline
        outcome_matrix = np.zeros((n_impl, n_actions))
        
        # Fit separate causal forest for each action vs baseline
        for i, action in enumerate(unique_actions):
            if action == 0:
                # Baseline: predict outcome under action 0 using regression forest
                from sklearn.ensemble import RandomForestRegressor
                
                X_baseline = x_mat[D_vec == 0]
                Y_baseline = y_vec[D_vec == 0]
                
                if len(X_baseline) == 0:
                    print(f"    Warning: No samples for baseline action 0")
                    outcome_matrix[:, i] = np.mean(y_vec)
                    continue
                
                rforest = RandomForestRegressor(
                    n_estimators=n_estimators_nuisance, 
                    min_samples_leaf=min_samples_leaf,
                    max_depth=max_depth,
                    random_state=42
                )
                rforest.fit(X_baseline, Y_baseline)
                
                mu_0 = rforest.predict(X_impl)
                outcome_matrix[:, i] = mu_0
                
            else:
                # Create binary indicator: D_binary = 1 if D==action, 0 if D==0
                mask = (D_vec == action) | (D_vec == 0)
                
                X_binary = x_mat[mask]
                Y_binary = y_vec[mask]
                D_binary_filtered = (D_vec[mask] == action).astype(float)
                
                if len(X_binary) == 0:
                    print(f"    Warning: No samples for action {action} vs baseline, using baseline")
                    outcome_matrix[:, i] = outcome_matrix[:, 0]
                    continue
                
                # Fit causal forest for action a vs baseline 0
                cforest_a = CausalForestDML(
                    model_y=RandomForestRegressor(
                        n_estimators=n_estimators_nuisance, 
                        min_samples_leaf=min_samples_leaf, 
                        random_state=42
                    ),
                    model_t=RandomForestRegressor(
                        n_estimators=n_estimators_nuisance, 
                        min_samples_leaf=min_samples_leaf, 
                        random_state=42
                    ),
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    max_depth=max_depth,
                    random_state=42,
                    verbose=0
                )
                
                cforest_a.fit(Y=Y_binary, T=D_binary_filtered, X=X_binary)
                
                # Predict treatment effect τ̂_a(x)
                tau_a = cforest_a.effect(X_impl).flatten()
                
                # Predicted outcome under action a = baseline + treatment effect
                outcome_matrix[:, i] = outcome_matrix[:, 0] + tau_a
        
        # Recommend action with highest predicted outcome
        seg_labels_impl = np.argmax(outcome_matrix, axis=1).astype(int)
        action_identity = unique_actions.astype(int)
    
    return seg_labels_impl, action_identity
