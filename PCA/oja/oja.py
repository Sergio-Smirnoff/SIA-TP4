
import numpy as np
import logging as log

class OjaPCA:
    """
    Implementación de la Regla de Oja para calcular la primera componente principal.
    Converge al autovector correspondiente al mayor autovalor de la matriz de covarianzas.
    """
    def __init__(self, learning_rate=0.001, n_epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        
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
            y = np.dot(X, self.w)
            updates = y[:, np.newaxis] * (X - y[:, np.newaxis] * self.w)
            self.w += self.learning_rate * np.mean(updates, axis=0)
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


