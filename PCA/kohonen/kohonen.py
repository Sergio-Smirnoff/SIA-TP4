import numpy as np
import logging as log

class Kohonen:
    """
    Implementación de la Red de Kohonen.
    """
    def __init__(self, k, data, init_r=1, init_learning_rate=0.01, epochs_rate=1000, similarity="euclidean", replace=False):
        self.data = data
        self.k = k
        # Inicializamos con ejemplos al azar del conjunto de entrenamiento
        idx = np.random.choice(len(self.data), size=k*k, replace=replace)
        self.weights = self.data[idx].reshape((k, k, self.data.shape[1]))
        self.r = init_r
        self.learning_rate = init_learning_rate
        self.n_epochs = epochs_rate*(k**2)
        if similarity == "euclidean":
            self.calculate_similarity = self.calculate_euclidean
        elif similarity == "exponential":
            self.calculate_similarity = self.calculate_exponential
        else:
            raise ValueError("similarity debe ser 'euclidean' o 'exponential'")
        
    def calculate_euclidean(self, x):
        distances = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(distances), (self.k, self.k))
    
    def calculate_exponential(self, x):
        x_norm = x / (np.linalg.norm(x) + 1e-9)
        weights_norm = self.weights / (np.linalg.norm(self.weights, axis=2, keepdims=True) + 1e-9)
        distances = np.linalg.norm(weights_norm - x_norm, axis=2)
        similarities = np.exp(-distances**2)
        return np.unravel_index(np.argmax(similarities), (self.k, self.k))

    def get_best_neuron(self, x):
        return self.calculate_similarity(x)
    
    def update_weights(self, x, idx):
        """
        Regla de Kohonen:
        Se actualiza el vecindario:
        N_k(i) = {n/||n - n_k|| < R(i)}
        de forma que:
        w_j(i+1) = w_j(t) + η(t) * (x - w(t))
        """
        for i in range(self.k):
            for j in range(self.k):
                distance_to_best_neuron = np.sqrt((i - idx[0])**2 + (j - idx[1])**2)
                # Buscamos si es vecina
                if distance_to_best_neuron < self.r:
                    # Actualizamos pesos
                    self.weights[i, j] += self.learning_rate * (x - self.weights[i, j])

    def train(self):
        log.info("Training Kohonen...")

        for epoch in range(self.n_epochs):
            # Seleccionamos un registro
            x=self.data[np.random.choice(len(self.data))]

            # Buscamos la neurona ganadora
            n_idx = self.get_best_neuron(x)

            # Actualizar los pesos
            self.update_weights(x, n_idx)

            # Reducimos η y el radio
            self.learning_rate = self.learning_rate * (1 - (epoch/self.n_epochs) )
            if self.r != 1:
                self.r = self.r * (1 - (epoch/self.n_epochs) )
                if self.r < 1:
                    self.r = 1
            if epoch % (self.n_epochs // 10) == 0:
                log.debug(f"Epoch {epoch}/{self.n_epochs}, lr={self.learning_rate:.4f}, r={self.r:.3f}")
        
        log.info("Training Kohonen completed")

    def predict(self, x):
        return np.array([self.get_best_neuron(xi) for xi in x])
    
    
        """
        Calcula la U-Matrix (Unified Distance Matrix) para visualizar las distancias
        entre neuronas vecinas.
        """
        k = self.k
        weights = self.weights
        u_matrix = np.zeros((k, k))

        for i in range(self.k):
            for j in range(self.k):
                distance_to_neighbor = np.sqrt((i - neighbor[0])**2 + (j - idx[1])**2)
                # Buscamos si es vecina
                if distance_to_best_neuron < self.r:
                    # Actualizamos pesos
                    self.weights[i, j] += self.learning_rate * (x - self.weights[i, j])

        for i in range(k):
            for j in range(k):
                current_weight = weights[i, j]
                distances_sum = 0
                neighbors_count = 0

                # Vecino de arriba
                if i > 0:
                    distances_sum += np.linalg.norm(current_weight - weights[i - 1, j])
                    neighbors_count += 1
                # Vecino de abajo
                if i < k - 1:
                    distances_sum += np.linalg.norm(current_weight - weights[i + 1, j])
                    neighbors_count += 1
                # Vecino de la izquierda
                if j > 0:
                    distances_sum += np.linalg.norm(current_weight - weights[i, j - 1])
                    neighbors_count += 1
                # Vecino de la derecha
                if j < k - 1:
                    distances_sum += np.linalg.norm(current_weight - weights[i, j + 1])
                    neighbors_count += 1
                
                # Calculamos la distancia promedio
                if neighbors_count > 0:
                    u_matrix[i, j] = distances_sum / neighbors_count
        
        return u_matrix