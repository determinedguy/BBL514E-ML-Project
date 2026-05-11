import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def tune_random_forest(X_train, y_train, random_state: int):
    # Switching from GridSearch to RandomizedSearch due to long (exhausting) runtime
    print("\nTuning Random Forest (Randomized Search)...")
    
    # 1. Initialize the base model
    rf_base = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    
    # 2. Define the Hyperparameter
    # n_estimators: Number of bootstrap trees (T)
    # max_depth: Prevents the deep-tree overfitting we saw in the baseline
    # min_samples_split: Requires a node to have at least N samples before splitting
   # We can even expand the grid now because we won't test every single one!
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # 3. Setup Random Search with 5-fold Cross-Validation
    # We use n_jobs=-1 to use all CPU cores, making this run much faster on Lightning
    print(f"Running Random Search (5-fold CV) with {len(param_dist['n_estimators']) * len(param_dist['max_depth']) * len(param_dist['min_samples_split'])} combinations...")
    
    # Start Timer
    start_time = time.time()

    # n_iter=10 means it will randomly pick ONLY 10 combinations to try,
    # reducing your total models trained from 135 down to just 30!
    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=10,          
        cv=5,
        scoring='f1',       
        n_jobs=-1,
        random_state=random_state,
        verbose=2           # Set to 2 so it prints its progress live!
    )
    
    # 4. Fit the Random Search
    random_search.fit(X_train, y_train)

    # End Timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    mins, secs = divmod(elapsed_time, 60)
    
    print(f"\n[DIAGNOSTIC] Tuning completed in {int(mins)}m {int(secs)}s")    
    print(f"Best Random Forest Parameters: {random_search.best_params_}")
    print(f"Best Cross-Validation F1-Score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_