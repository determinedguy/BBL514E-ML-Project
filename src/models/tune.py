import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample

def tune_random_forest(X_train, y_train, random_state: int):
    print("\n--- Tuning Random Forest (10% Sub-Sampled Randomized Search) ---")
    start_time = time.time()
    
    # 1. Downsample to 10% for hyperparameter tuning to prevent memory exhaustion
    print("Downsampling SMOTE data to 10% for rapid tuning...")
    X_tune, y_tune = resample(
        X_train, y_train, 
        n_samples=int(len(X_train) * 0.10), 
        stratify=y_train, 
        random_state=random_state
    ) # type: ignore
    
    # 2. Initialize the base model and define the hyperparameter
    rf_base = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }

    # 3. Setup RandomizedSearch with 5-fold Cross-Validation
    # We use n_jobs=-1 to use all CPU cores, making this run much faster on Lightning
    print(f"Running Random Search (5-fold CV) with {len(param_dist['n_estimators']) * len(param_dist['max_depth']) * len(param_dist['min_samples_split'])} combinations...")
    
    # Start Timer
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=10,          
        cv=5,               
        scoring='f1',       
        n_jobs=-1,
        random_state=random_state,
        verbose=1 # Reduced verbosity so it doesn't flood the terminal
    )
    
    # 4. Fit the search ONLY on the 10% sub-sample
    print("Fitting RandomizedSearchCV on 10% sub-sample...")
    random_search.fit(X_tune, y_tune)
    
    print(f"\nBest Parameters Found: {random_search.best_params_}")
    
    # 5. Train the final, optimized model on the FULL 100% dataset
    print("Training final Random Forest with best parameters on FULL dataset...")
    best_rf = RandomForestClassifier(
        **random_search.best_params_, 
        random_state=random_state, 
        n_jobs=-1
    )
    best_rf.fit(X_train, y_train)
    
    # Diagnostics
    mins, secs = divmod(time.time() - start_time, 60)
    print(f"[DIAGNOSTIC] Tuning & Training completed in {int(mins)}m {int(secs)}s")
    
    return best_rf