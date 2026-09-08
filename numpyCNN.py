import os
import numpy as np
from PIL import Image

# =====================================================================
# 1. HYPERPARAMETERS & SETTINGS
# =====================================================================
INPUT_HEIGHT = 32      # Kept small (32x32) for fast NumPy training loops
INPUT_WIDTH = 32       
IN_CHANNELS = 3        # RGB channels
NUM_CLASSES = 4        # 4 groups
SAMPLES_PER_CLASS = 100

LEARNING_RATE = 0.01
EPOCHS = 5
BATCH_SIZE = 32


# =====================================================================
# 2. DATASET LOADER (READS 4 FOLDERS)
# =====================================================================
data, labels = [], []
print("--- Loading Dataset from Folders ---")
for class_idx in range(NUM_CLASSES):
    folder_name = f"group_{class_idx}"
    if os.path.exists(folder_name):
        image_files = [f for f in sorted(os.listdir(folder_name)) 
                       if f.lower().endswith(('png', 'jpg', 'jpeg'))][:SAMPLES_PER_CLASS]
        for img_file in image_files:
            img_path = os.path.join(folder_name, img_file)
            img = Image.open(img_path).convert('RGB').resize((INPUT_WIDTH, INPUT_HEIGHT))
            data.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(class_idx)
        print(f"Loaded {len(image_files)} images from '{folder_name}/'")
    else:
        print(f"Warning: Folder '{folder_name}/' not found. Using synthetic fallback for test.")

if len(data) > 0:
    X_train = np.array(data)
    y_train = np.array(labels)
else:
    # Fallback dummy data if folders are empty during testing
    X_train = np.random.randn(400, INPUT_HEIGHT, INPUT_WIDTH, IN_CHANNELS)
    y_train = np.repeat(np.arange(NUM_CLASSES), 100)


# =====================================================================
# 3. NETWORK LAYERS & BACKPROPAGATION COMPONENTS
# =====================================================================

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.kernel_size = kernel_size
        self.filters = np.random.randn(out_channels, kernel_size, kernel_size, in_channels) * 0.1
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
                    # Gradient accumulation for filters
                    d_filters[f] += np.sum(patch * dout[:, i, j, f][:, np.newaxis, np.newaxis, np.newaxis], axis=0)
                    # Gradient backpropagation to previous layer (d_X)
                    d_X[:, i:i+KK, j:j+KK, :] += self.filters[f] * dout[:, i, j, f][:, np.newaxis, np.newaxis, np.newaxis]
                    
        # Gradient Descent Weight Update
        self.filters -= lr * d_filters / N
        self.biases -= lr * d_biases / N
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
                # Route gradient back to the max value locations
                max_val = np.max(patch, axis=(1, 2, keepdims=True)) if hasattr(np, 'newaxis') else np.max(patch, axis=(1,2))
                mask = (patch == np.max(patch, axis=(1, 2), keepdims=True))
                d_X[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps, :] += mask * dout[:, i:i+1, j:j+1, :] # Simplified routing
        return d_X

class Dense:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros((1, output_dim))

    def forward(self, X):
        self.X = X # Shape: (N, input_dim)
        return np.dot(X, self.weights) + self.biases

    def backward(self, dout, lr):
        N = self.X.shape[0]
        d_weights = np.dot(self.X.T, dout)
        d_biases = np.sum(dout, axis=0, keepdims=True)
        d_X = np.dot(dout, self.weights.T)
        
        # Gradient Descent Update
        self.weights -= lr * d_weights / N
        self.biases -= lr * d_biases / N
        return d_X

def softmax_cross_entropy_loss(logits, y_true):
    # Softmax calculation for stability
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    
    N = logits.shape[0]
    core_loss = -np.log(probs[np.arange(N], y_true) + 1e-9)
    loss = np.mean(core_loss)
    
    # Gradient of Softmax + Cross Entropy combined
    d_logits = probs.copy()
    d_logits[np.arange(N), y_true] -= 1
    d_logits /= N
    
    return loss, d_logits, probs


# =====================================================================
# 4. TRAINING LOOP
# =====================================================================
if __name__ == "__main__":
    print("\n--- Initializing Trainable CNN Architecture ---")
    
    # Instantiate Layers
    conv = Conv2D(in_channels=IN_CHANNELS, out_channels=4, kernel_size=3)
    relu = ReLU()
    pool = MaxPool2D(pool_size=2)
    
    # Calculate flattened size dynamically after Conv and Pool
    # For 32x32 -> Conv(3x3, valid) -> 30x30 -> Pool(2x2) -> 15x15 -> 4 filters = 15 * 15 * 4 = 900
    flatten_dim = (INPUT_HEIGHT - 2) // 2 * (INPUT_WIDTH - 2) // 2 * 4
    dense = Dense(input_dim=flatten_dim, output_dim=NUM_CLASSES)

    print(f"Training on {X_train.shape[0]} samples for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        # Shuffle dataset each epoch
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        correct_preds = 0
        
        # Mini-batch training loop
        for i in range(0, X_train.shape[0], BATCH_SIZE):
            X_batch = X_shuffled[i:i+BATCH_SIZE]
            y_batch = y_shuffled[i:i+BATCH_SIZE]
            
            # 1. Forward Pass
            out_conv = conv.forward(X_batch)
            out_relu = relu.forward(out_conv)
            out_pool = pool.forward(out_relu)
            
            # Flatten for Dense layer
            N_b, H_b, W_b, C_b = out_pool.shape
            out_flat = out_pool.reshape(N_b, -1)
            
            logits = dense.forward(out_flat)
            
            # 2. Loss & Softmax
            loss, d_logits, probs = softmax_cross_entropy_loss(logits, y_batch)
            epoch_loss += loss * N_b
            
            # Track accuracy
            preds = np.argmax(probs, axis=1)
            correct_preds += np.sum(preds == y_batch)
            
            # 3. Backward Pass (Backpropagation)
            d_flat = dense.backward(d_logits, LEARNING_RATE)
            d_pool = d_flat.reshape(N_b, H_b, W_b, C_b)
            d_relu = pool.backward(d_pool)
            d_conv = relu.backward(d_relu)
            _ = conv.backward(d_conv, LEARNING_RATE)
            
        avg_loss = epoch_loss / X_train.shape[0]
        accuracy = (correct_preds / X_train.shape[0]) * 100
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    print("\n--- Training Completed Successfully! ---")