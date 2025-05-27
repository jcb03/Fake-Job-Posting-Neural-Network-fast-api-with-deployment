import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

class ImbalanceAwareNeuralNetwork:
    """Neural network with class imbalance handling capabilities"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64], learning_rate=0.0001, 
                 dropout_rate=0.63, class_weights=None, focal_loss=False):
        """
        Initialize neural network with imbalance handling
        
        Parameters:
        input_size: number of input features
        hidden_sizes: list of hidden layer sizes
        learning_rate: learning rate for optimization
        dropout_rate: dropout probability for regularization
        class_weights: dictionary of class weights for weighted loss
        focal_loss: whether to use focal loss for imbalanced data
        """
        # Removed print statements for deployment
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.class_weights = class_weights
        self.focal_loss = focal_loss
        self.layers = []
        
        # Initialize weights and biases
        layer_sizes = [input_size] + hidden_sizes + [1]
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU layers
            if i < len(layer_sizes) - 2:
                weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            else:
                weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1.0 / layer_sizes[i])
            
            bias = np.zeros((1, layer_sizes[i+1]))
            
            self.layers.append({
                'weight': weight,
                'bias': bias,
                'activation': None,
                'z': None,
                'dropout_mask': None
            })
    
    def sigmoid(self, z):
        """Sigmoid activation with numerical stability"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def leaky_relu(self, z, alpha=0.01):
        """Leaky ReLU activation"""
        return np.where(z > 0, z, alpha * z)
    
    def leaky_relu_derivative(self, z, alpha=0.01):
        """Derivative of Leaky ReLU"""
        return np.where(z > 0, 1, alpha)
    
    def apply_dropout(self, x, training=True):
        """Apply dropout regularization"""
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
            return x * mask, mask
        else:
            return x, None
    
    def forward_propagation(self, X, training=True):
        """Forward propagation with dropout"""
        current_input = X
        
        for i, layer in enumerate(self.layers):
            # Linear transformation
            z = np.dot(current_input, layer['weight']) + layer['bias']
            layer['z'] = z
            
            if i == len(self.layers) - 1:  # Output layer
                activation = self.sigmoid(z)
            else:  # Hidden layers
                activation = self.leaky_relu(z)
                if training:
                    activation, dropout_mask = self.apply_dropout(activation, training)
                    layer['dropout_mask'] = dropout_mask
            
            layer['activation'] = activation
            current_input = activation
        
        return current_input
    
    def compute_weighted_loss(self, y_true, y_pred):
        """Compute weighted binary cross-entropy loss"""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        if self.focal_loss:
            # Focal loss for hard examples
            alpha = 0.25
            gamma = 2.0
            
            ce_loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
            focal_weight = alpha * (1 - p_t) ** gamma
            focal_loss = focal_weight * ce_loss
            
            loss = np.mean(focal_loss)
        else:
            # Standard weighted cross-entropy
            if self.class_weights is not None:
                # Fixed: Apply class weights correctly
                weights = np.where(y_true == 1, self.class_weights[1], self.class_weights[0])
                ce_loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
                weighted_loss = weights * ce_loss
                loss = np.mean(weighted_loss)
            else:
                # Standard cross-entropy
                loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # L2 regularization
        l2_reg = 0
        for layer in self.layers:
            l2_reg += np.sum(layer['weight'] ** 2)
        l2_reg *= 0.05
        
        return loss + l2_reg
    
    def backward_propagation(self, X, y_true, y_pred):
        """Backward propagation with weighted gradients"""
        m = X.shape[0]  # Fixed: Get the correct batch size
        gradients = []
        
        # Compute weighted gradient for output layer
        if self.class_weights is not None and not self.focal_loss:
            # Fixed: Apply class weights correctly
            weights = np.where(y_true == 1, self.class_weights[1], self.class_weights[0])
            dz = (y_pred - y_true.reshape(-1, 1)) * weights.reshape(-1, 1)
        else:
            dz = y_pred - y_true.reshape(-1, 1)
        
        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            if i == 0:
                prev_activation = X
            else:
                prev_activation = self.layers[i-1]['activation']
            
            dw = np.dot(prev_activation.T, dz) / m
            db = np.mean(dz, axis=0, keepdims=True)
            
            # Add L2 regularization
            dw += 0.05 * layer['weight']
            
            gradients.insert(0, {'dw': dw, 'db': db})
            
            if i > 0:
                da_prev = np.dot(dz, layer['weight'].T)
                if self.layers[i-1]['dropout_mask'] is not None:
                    da_prev *= self.layers[i-1]['dropout_mask']
                dz = da_prev * self.leaky_relu_derivative(self.layers[i-1]['z'])
        
        return gradients
    
    def update_parameters(self, gradients):
        """Update parameters using gradient descent"""
        for i, (layer, grad) in enumerate(zip(self.layers, gradients)):
            layer['weight'] -= self.learning_rate * grad['dw']
            layer['bias'] -= self.learning_rate * grad['db']
    
    def train(self, X, y, epochs=200, batch_size=64, validation_split=0.2, verbose=True):
        """Enhanced training with imbalance-aware monitoring"""
        # Removed print statements for deployment, but keep essential ones
        
        # Stratified split for validation
        val_size = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        
        X_train, X_val = X[indices[val_size:]], X[indices[:val_size]]
        y_train, y_val = y[indices[val_size:]], y[indices[:val_size]]
        
        # Training history
        history = {
            'train_losses': [], 'val_losses': [],
            'train_accuracies': [], 'val_accuracies': [],
            'train_f1': [], 'val_f1': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': []
        }
        
        best_val_f1 = 0
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Shuffle training data
            train_indices = np.random.permutation(len(X_train))
            epoch_train_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_indices = train_indices[i:i+batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                # Forward propagation
                y_pred = self.forward_propagation(X_batch, training=True)
                
                # Compute loss
                batch_loss = self.compute_weighted_loss(y_batch, y_pred)
                epoch_train_loss += batch_loss
                num_batches += 1
                
                # Backward propagation
                gradients = self.backward_propagation(X_batch, y_batch, y_pred)
                
                # Update parameters
                self.update_parameters(gradients)
            
            # Calculate epoch metrics
            avg_train_loss = epoch_train_loss / num_batches
            
            # Validation metrics
            y_val_pred = self.forward_propagation(X_val, training=False)
            val_loss = self.compute_weighted_loss(y_val, y_val_pred)
            
            # Calculate comprehensive metrics
            train_pred = self.forward_propagation(X_train, training=False)
            
            # Training metrics
            train_acc = np.mean((train_pred > 0.5) == y_train.reshape(-1, 1))
            train_f1 = f1_score(y_train, (train_pred > 0.5).astype(int))
            train_precision = precision_score(y_train, (train_pred > 0.5).astype(int))
            train_recall = recall_score(y_train, (train_pred > 0.5).astype(int))
            
            # Validation metrics
            val_acc = np.mean((y_val_pred > 0.5) == y_val.reshape(-1, 1))
            val_f1 = f1_score(y_val, (y_val_pred > 0.5).astype(int))
            val_precision = precision_score(y_val, (y_val_pred > 0.5).astype(int))
            val_recall = recall_score(y_val, (y_val_pred > 0.5).astype(int))
            
            # Store history
            history['train_losses'].append(avg_train_loss)
            history['val_losses'].append(val_loss)
            history['train_accuracies'].append(train_acc)
            history['val_accuracies'].append(val_acc)
            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)
            history['train_precision'].append(train_precision)
            history['val_precision'].append(val_precision)
            history['train_recall'].append(train_recall)
            history['val_recall'].append(val_recall)
            
            # Early stopping based on F1 score (better for imbalanced data)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Reduced verbosity for deployment
            if verbose and epoch % 25 == 0:
                print(f"   Epoch {epoch:3d}: Loss: {avg_train_loss:.4f}/{val_loss:.4f}, "
                      f"Acc: {train_acc:.4f}/{val_acc:.4f}, "
                      f"F1: {train_f1:.4f}/{val_f1:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"   Early stopping at epoch {epoch} (best val F1: {best_val_f1:.4f})")
                break
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward_propagation(X, training=False)
        return (y_pred > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.forward_propagation(X, training=False).flatten()
    
    def get_model_info(self):
        """Get model architecture information for debugging"""
        info = {
            'num_layers': len(self.layers),
            'layer_sizes': [layer['weight'].shape for layer in self.layers],
            'total_parameters': sum(layer['weight'].size + layer['bias'].size for layer in self.layers),
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'has_class_weights': self.class_weights is not None,
            'focal_loss': self.focal_loss
        }
        return info
