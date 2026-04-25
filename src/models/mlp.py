# Multilayer Perceptron
from sklearn.neural_network import MLPClassifier
from src.models.export import save_model
import time

def train_fast_mlp(X_train, y_train, random_state: int):
    print("\n--- Training Fast MLP Prototype (scikit-learn) ---")
    
    # Start timer
    start_time = time.time()
    
    # 1. Initialize the Neural Network
    # hidden_layer_sizes=(64, 32) creates two hidden layers. 
    # early_stopping=True prevents it from training too long if accuracy stops improving.
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',       # Matches your proposal: h = ReLU(W1*X + b1)
        solver='adam',
        early_stopping=True,     # Forces it to finish quickly!
        validation_fraction=0.1, # Uses 10% of training data to monitor early stopping
        random_state=random_state,
        verbose=True             # Prints epoch progress to the terminal
    )
    
    # 2. Fit the model on the full SMOTE dataset
    print("Fitting MLP to the full training dataset...")
    mlp_model.fit(X_train, y_train)
    
    # End timer
    mins, secs = divmod(time.time() - start_time, 60)
    print(f"\n[DIAGNOSTIC] MLP Training completed in {int(mins)}m {int(secs)}s")
    
    return mlp_model