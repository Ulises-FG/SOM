import numpy as np
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, dim_entrada, dim_salida, lr=0.1):
        self.dim_entrada = dim_entrada
        self.dim_salida = dim_salida
        self.lr = lr
        # Inicializa los pesos aleatorios de la matriz de pesos de la red SOM.
        self.pesos = np.random.rand(dim_salida[0], dim_salida[1], dim_entrada)
        
    def entrenar(self, datos, epochs,max_epochs):
        for epoch in range(epochs):
            for dato in datos:
                # Encuentra la neurona ganadora o BMU.
                bmu = self.obtener_bmu(dato)
                # Actualiza los pesos de las neuronas en la red SOM.
                self.actualizar_pesos(bmu, dato, epoch, max_epochs)
                
    def obtener_bmu(self, dato):
        # Calcula la distancia euclidiana entre cada neurona de la red SOM y el vector de entrada.
        distancias = np.sqrt(np.sum((self.pesos - dato)**2, axis=2))
        # Encuentra la neurona cuyo peso está más cerca del vector de entrada.
        bmu_pos = np.unravel_index(np.argmin(distancias), distancias.shape)
        return bmu_pos
        
    def actualizar_pesos(self, bmu, dato, epoch, max_epochs):
        # Calcula la tasa de aprendizaje para la regla de aprendizaje de Kohonen.
        lr = self.lr * (1 - epoch / max_epochs)
        for i in range(self.dim_salida[0]):
            for j in range(self.dim_salida[1]):
                # Calcula la distancia euclidiana entre la neurona actual y la neurona ganadora.
                dist = np.sqrt((i-bmu[0])**2 + (j-bmu[1])**2)
                # Calcula la influencia de la neurona ganadora sobre la neurona actual.
                influencia = np.exp(-(dist**2) / (2*(lr**2)))
                # Actualiza los pesos de la neurona actual usando la regla de aprendizaje de Kohonen.
                self.pesos[i,j,:] += influencia * lr * (dato - self.pesos[i,j,:])

    def visualizar(self, datos):
        # Crea una imagen en blanco de la red SOM.
        plt.imshow(np.zeros(self.dim_salida))
        for dato in datos:
            # Encuentra la neurona ganadora para el punto de datos actual.
            bmu = self.obtener_bmu(dato)
            # Grafica el punto de datos en la posición de la neurona ganadora.
            plt.scatter(bmu[1], bmu[0], c=dato)
        plt.show()

# Genera datos aleatorios de entrada.
datos = np.random.rand(100, 3)
# Crea una instancia de la red SOM.
# dim_entrada es el número de características de los datos de entrada (o dimensiones).
# dim_salida crea una cuadricula de 3x2 neuronas.
som = SOM(dim_entrada=3, dim_salida=(3,2), lr=0.1)
# Entrena la red SOM con los datos de entrada.
som.entrenar(datos, epochs=100, max_epochs=100)
# Visualiza los datos de entrada en la red SOM después del entrenamiento.
som.visualizar(datos)
