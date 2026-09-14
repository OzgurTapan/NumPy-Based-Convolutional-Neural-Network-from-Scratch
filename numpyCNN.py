import os
import numpy as np
from PIL import Image

# =====================================================================
# 1. HYPERPARAMETERS & SETTINGS
# =====================================================================
# --- Dataset & Input Dimensions ---
INPUT_HEIGHT = 32      # Target image height for resizing during load
INPUT_WIDTH = 32       # Target image width for resizing during load
IN_CHANNELS = 3        # Color channels (3 for RGB, 1 for Grayscale)
NUM_CLASSES = 4        # Total distinct categories/folders to classify

# --- Model Architecture Settings ---
CONV_OUT_CHANNELS = 4  # Number of feature detectors (filters) in Conv2D
CONV_KERNEL = 3        # Sliding window size (3x3) for Conv2D
POOL_SIZE = 2          # Downsampling window size (2x2) for MaxPool

# --- Training Parameters ---
LEARNING_RATE = 0.01   # Step size for weight updates during backprop
EPOCHS = 10            # Total complete passes over the entire dataset
BATCH_SIZE = 32        # Number of images processed simultaneously


# =====================================================================
# 2. DATASET LOADER FOR SEPARATE TRAIN & TEST FOLDERS
# =====================================================================
def load_dataset_from_directory(base_dir):
    """Scans base_dir ('train' or 'test') for group_0 to group_3 subfolders."""
    data, labels = [], []
    
    if not os.path.exists(base_dir):
        print(f"Warning: Directory '{base_dir}/' not found!")
        return np.array(data), np.array(labels)
        
    print(f"--- Loading Data from '{base_dir}/' ---")
    
    for class_idx in range(NUM_CLASSES):
        folder_path = os.path.join(base_dir, f"group_{class_idx}")
        
        if os.path.exists(folder_path):
            image_files = [f for f in sorted(os.listdir(folder_path)) 
                           if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                
                # Dynamically choose color mode
                mode = 'RGB' if IN_CHANNELS == 3 else 'L'
                img = Image.open(img_path).convert(mode).resize((INPUT_WIDTH, INPUT_HEIGHT))
                img_arr = np.array(img, dtype=np.float32) / 255.0
                
                # Add channel dimension for Grayscale (H, W) -> (H, W, 1)
                if IN_CHANNELS == 1:
                    img_arr = np.expand_dims(img_arr, axis=-1)
                    
                data.append(img_arr)
                labels.append(class_idx)
                
            print(f"Loaded {len(image_files)} images from '{base_dir}/group_{class_idx}/'")
        else:
            print(f"Warning: Folder '{folder_path}' not found!")
            
    return np.array(data), np.array(labels)


# =====================================================================
# 3. NETWORK LAYERS & BACKPROPAGATION COMPONENTS
# =====================================================================
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.kernel_size = kernel_size
        self.filters = np.random.randn(out_channels, kernel_size, kernel_size, in_channels) * np.sqrt(2.0 / (kernel_size**2 * in_channels))
        self.biases = np.zeros(out_channels)
        
    def forward(self, X):
        self.X = X
        N, H, W, C = X.shape
        F, KK, _, _ = self.filters.shape
        out_h, out_w = H - KK + 1, W - KK + 1
        self.out_h, self.out_w = out_h, out_w
        
        out = np.zeros((N, out_h, out_w, F))
        for f in range(F):
            for i in range(out_h):
                for j in range(out_w):
                    patch = X[:, i:i+KK, j:j+KK, :]
                    out[:, i, j, f] = np.sum(patch * self.filters[f], axis=(1, 2, 3)) + self.biases[f]
        return out

    def backward(self, dout, lr):
        N, H, W, C = self.X.shape
        F, KK, _, _ = self.filters.shape
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_X = np.zeros_like(self.X)
        
        for f in range(F):
            d_biases[f] = np.sum(dout[:, :, :, f])
            for i in range(self.out_h):
                for j in range(self.out_w):
                    patch = self.X[:, i:i+KK, j:j+KK, :]
                    d_filters[f] += np.sum(patch * dout[:, i, j, f][:, np.newaxis, np.newaxis, np.newaxis], axis=0)
                    d_X[:, i:i+KK, j:j+KK, :] += self.filters[f] * dout[:, i, j, f][:, np.newaxis, np.newaxis, np.newaxis]
                    
        self.filters -= lr * d_filters
        self.biases -= lr * d_biases
        return d_X

class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dout):
        return dout * (self.X > 0)

class MaxPool2D:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, X):
        self.X = X
        N, H, W, C = X.shape
        ps = self.pool_size
        out_h, out_w = H // ps, W // ps
        
        out = np.zeros((N, out_h, out_w, C))
        for i in range(out_h):
            for j in range(out_w):
                patch = X[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps, :]
                out[:, i, j, :] = np.max(patch, axis=(1, 2))
        return out

    def backward(self, dout):
        N, H, W, C = self.X.shape
        ps = self.pool_size
        d_X = np.zeros_like(self.X)
        out_h, out_w = dout.shape[1], dout.shape[2]
        
        for i in range(out_h):
            for j in range(out_w):
                patch = self.X[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps, :]
                mask = (patch == np.max(patch, axis=(1, 2), keepdims=True))
                
                # <--- GRADIENT DUPLICATION FIX --->
                # Distributes the gradient equally among all pixels in the patch that tied for the maximum value 
                # (e.g. if two pixels tied for max, each receives 50% of the gradient). This prevents the total 
                # backward gradient from exceeding 100% of the incoming dout gradient.
                mask = mask / np.sum(mask, axis=(1, 2), keepdims=True)
                # <-------------------------------->
                
                d_X[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps, :] += mask * dout[:, i:i+1, j:j+1, :]
        return d_X

class Dense:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.biases = np.zeros((1, output_dim))

    def forward(self, X):
        self.X = X 
        return np.dot(X, self.weights) + self.biases

    def backward(self, dout, lr):
        d_weights = np.dot(self.X.T, dout)
        d_biases = np.sum(dout, axis=0, keepdims=True)
        d_X = np.dot(dout, self.weights.T)
        
        self.weights -= lr * d_weights
        self.biases -= lr * d_biases
        return d_X

def softmax_cross_entropy_loss(logits, y_true):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    
    N = logits.shape[0]
    core_loss = -np.log(probs[np.arange(N), y_true] + 1e-9)
    loss = np.mean(core_loss)
    
    d_logits = probs.copy()
    d_logits[np.arange(N), y_true] -= 1
    d_logits /= N  
    
    return loss, d_logits, probs


# =====================================================================
# 4. SAVE & LOAD UTILITIES
# =====================================================================
def save_weights(filepath, conv_layer, dense_layer):
    """Saves model parameters to a compressed .npz file."""
    np.savez_compressed(
        filepath,
        conv_filters=conv_layer.filters,
        conv_biases=conv_layer.biases,
        dense_weights=dense_layer.weights,
        dense_biases=dense_layer.biases
    )
    print(f"\n[INFO] Model weights successfully saved to {filepath}.npz")

def load_weights(filepath, conv_layer, dense_layer):
    """Loads model parameters from a .npz file."""
    if not filepath.endswith('.npz'):
        filepath += '.npz'
        
    if not os.path.exists(filepath):
        print(f"\n[WARNING] Weight file '{filepath}' not found. Initializing from scratch.")
        return False
        
    data = np.load(filepath)
    conv_layer.filters = data['conv_filters']
    conv_layer.biases = data['conv_biases']
    dense_layer.weights = data['dense_weights']
    dense_layer.biases = data['dense_biases']
    print(f"\n[INFO] Model weights successfully loaded from {filepath}")
    return True


# =====================================================================
# 5. TRAINING AND EVALUATION PIPELINE
# =====================================================================
if __name__ == "__main__":
    
    X_train, y_train = load_dataset_from_directory("dataset/train")
    X_test, y_test = load_dataset_from_directory("dataset/test")
    
    if X_train.size == 0 or X_test.size == 0:
        print("\nError: Training or testing data is missing.")
        exit()

    print(f"\nTraining Set Shape: {X_train.shape}")
    print(f"Testing Set Shape:  {X_test.shape}\n")

    # Dynamic layer shape calculations
    conv_h = INPUT_HEIGHT - CONV_KERNEL + 1
    conv_w = INPUT_WIDTH - CONV_KERNEL + 1
    pool_h = conv_h // POOL_SIZE
    pool_w = conv_w // POOL_SIZE
    flatten_dim = pool_h * pool_w * CONV_OUT_CHANNELS

    # Initialize Layers
    print("--- Initializing CNN Layers ---")
    conv = Conv2D(in_channels=IN_CHANNELS, out_channels=CONV_OUT_CHANNELS, kernel_size=CONV_KERNEL)
    relu = ReLU()
    pool = MaxPool2D(pool_size=POOL_SIZE)
    dense = Dense(input_dim=flatten_dim, output_dim=NUM_CLASSES)

    # UNCOMMENT THE LINE BELOW TO RESUME TRAINING FROM SAVED WEIGHTS
    # load_weights("cnn_weights", conv, dense)

    def evaluate(X, y):
        correct_predictions = 0
        
        # Process the evaluation in batches to prevent Out-Of-Memory (OOM) crashes
        for i in range(0, X.shape[0], BATCH_SIZE):
            X_batch = X[i:i+BATCH_SIZE]
            y_batch = y[i:i+BATCH_SIZE]
            
            # Forward Pass
            out_conv = conv.forward(X_batch)
            out_relu = relu.forward(out_conv)
            out_pool = pool.forward(out_relu)
            out_flat = out_pool.reshape(X_batch.shape[0], -1)
            logits = dense.forward(out_flat)
            
            # Extract predictions directly from highest logit value
            preds = np.argmax(logits, axis=1)
            correct_predictions += np.sum(preds == y_batch)
            
        return (correct_predictions / X.shape[0]) * 100

    print(f"Starting training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        
        for i in range(0, X_train.shape[0], BATCH_SIZE):
            X_batch = X_shuffled[i:i+BATCH_SIZE]
            y_batch = y_shuffled[i:i+BATCH_SIZE]
            
            # Forward Pass
            out_conv = conv.forward(X_batch)
            out_relu = relu.forward(out_conv)
            out_pool = pool.forward(out_relu)
            
            N_b, H_b, W_b, C_b = out_pool.shape
            out_flat = out_pool.reshape(N_b, -1)
            logits = dense.forward(out_flat)
            
            # Loss Evaluation
            loss, d_logits, probs = softmax_cross_entropy_loss(logits, y_batch)
            epoch_loss += loss * N_b 
            
            # Backward Pass
            d_flat = dense.backward(d_logits, LEARNING_RATE)
            d_pool = d_flat.reshape(N_b, H_b, W_b, C_b)
            d_relu = pool.backward(d_pool)
            d_conv = relu.backward(d_relu)
            _ = conv.backward(d_conv, LEARNING_RATE)
            
        train_acc = evaluate(X_train, y_train)
        test_acc = evaluate(X_test, y_test)
        avg_loss = epoch_loss / X_train.shape[0]
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

    print("\n--- Training and Evaluation Completed! ---")
    
    # Save the weights after training finishes
    save_weights("cnn_weights", conv, dense)
