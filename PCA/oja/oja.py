
import numpy as np
import logging as log
import sklearn.preprocessing as StandardScaler 

class OjaPCA:
    """
    Implementación de la Regla de Oja para calcular la primera componente principal.
    Converge al autovector correspondiente al mayor autovalor de la matriz de covarianzas.
    """
    def __init__(self, learning_rate=0.01, n_epochs=10000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.w = None

        np.random.seed(42)
        
    def fit(self, X):
        """
        Entrenar usando la regla de Oja para obtener la primera componente principal.
        
        Regla de Oja:
        w += η * y * (x - y * w)
        
        donde y = x · w (producto interno)
        """
        n_samples, n_features = X.shape
        
        self.w = np.random.uniform(0, 1, n_features)
        self.w = self.w / np.linalg.norm(self.w)
        
        log.info("Training Oja's rule for PC1...")
        
        for epoch in range(self.n_epochs):
            O = np.dot(X, self.w)
            
            w_norm = np.linalg.norm(self.w)

            term1 = np.dot(X.T, O) / w_norm

            sum_term = np.dot(O,O)
            term2 = (sum_term * self.w) / (w_norm ** 3)

            delta_w = (term1 - term2) / n_samples
            self.w += self.learning_rate * delta_w

            self.w = self.w / np.linalg.norm(self.w)
            
            if (epoch + 1) % 20 == 0:
                log.debug(f"Epoch {epoch + 1}/{self.n_epochs} completed")
        
        log.info("Oja's rule training completed")
        return self
    
    def transform(self, X):
        """
        Proyectar los datos en la primera componente principal.
        """
        return np.dot(X, self.w.reshape(-1, 1))


