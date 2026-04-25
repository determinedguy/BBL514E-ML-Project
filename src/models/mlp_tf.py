import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import time

def build_and_train_tf(X_train, y_train, X_val, y_val, random_state: int):
    print("\n--- Training TensorFlow MLP (Strict Proposal Compliance) ---")
    tf.random.set_seed(random_state)
    
    # Architecture exactly matching your proposal
    model = models.Sequential([
        layers.InputLayer(input_shape=(X_train.shape[1],)),
        
        # h = ReLU(W1*X + b1)
        layers.Dense(64, activation='relu', name='Hidden_Layer_1_ReLU'),
        
        # y_hat = softmax(W2*h + b2)
        layers.Dense(2, activation='softmax', name='Output_Layer_Softmax') 
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Stop early if it stops learning, keep the best weights
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5,          
        restore_best_weights=True 
    )
    
    start_time = time.time()
    print("\nStarting overnight training. You can safely step away...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,          
        batch_size=256,     # The magic bullet to prevent RAM exhaustion
        callbacks=[early_stopping],
        verbose=1
    )
    
    mins, secs = divmod(time.time() - start_time, 60)
    print(f"\n[DIAGNOSTIC] TensorFlow Training completed in {int(mins)}m {int(secs)}s")
    
    return model