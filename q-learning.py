import time
import gymnasium as gym
import numpy as np

N = 1000 # Número de generaciones para mostrar información
desc = ["SFFF", "FHFH", "FFFH", "HFFG"]

# Entornos
slippery = True
training_env = gym.make('FrozenLake-v1', desc=desc, is_slippery=slippery)
rendering_env = gym.make('FrozenLake-v1', desc=desc, is_slippery=slippery, render_mode='human')

# Parámetros del entorno
num_estados = training_env.observation_space.n # 16
num_acciones = training_env.action_space.n # 4

# Al inicio la matriz se debe inicializar en 0 con num_estados filas y num_acciones columnas
q_table = np.zeros((num_estados, num_acciones)) 

tasa_aprendizaje = 0.15 # ALPHA
factor_descuento = 0.99 # GAMMA
generaciones = 30000 # EPISODIOS
listado_recompensas = []

# Epsilon-greedy
epsilon = 1.0 # Exploración inicial
epsilon_min = 0.05 # Epsilon mínimo
decaimiento = 0.9995  # Factor de decaimiento de epsilon
max_steps = 90 # Máximo de pasos por episodio

for generacion in range(generaciones):
    observation, _ = training_env.reset() # Reiniciar el entorno
    estado = observation # Estado inicial
    final = False # Variable para indicar si el episodio ha terminado
    recompensa_total = 0 # Recompensa total del episodio
    steps = 0 # Contador de pasos

    while not final and steps < max_steps: # Mientras no se alcance un estado terminal y no se exceda el máximo de pasos
        if np.random.rand() > epsilon: # Explorar o explotar
            accion = np.argmax(q_table[estado]) # Explotar: elegir la acción con el valor Q más alto
        else:
            accion = training_env.action_space.sample() #acción aleatoria entre 0 y 3

        nuevo_estado, recompensa, terminated, truncated, _ = training_env.step(accion) # Realizar la acción en el entorno
        final = terminated or truncated # El episodio ha terminado si se ha alcanzado un estado terminal o truncado

        q_table[estado][accion] += tasa_aprendizaje * (
            recompensa + factor_descuento * np.max(q_table[nuevo_estado]) - q_table[estado][accion]
            ) # Actualizar la Q-table con fórmula de Q-learning: Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))

        estado = nuevo_estado
        recompensa_total += recompensa
        steps += 1 # Incrementar el contador de pasos

    listado_recompensas.append(recompensa_total)

    epsilon = max(epsilon_min, epsilon * decaimiento) # Actualizar epsilon para la siguiente iteración
            
    # Mostrar info cada N generaciones
    if (generacion + 1) % N == 0: # Mostrar información cada N generaciones
        promedio = np.mean(listado_recompensas[-100:])
        print(f'Generación {generacion+1} - Promedio últimas 100 recompensas: {promedio:.4f} - Épsilon: {epsilon:.4f}')

pruebas = 4 if slippery else 1
for i in range(pruebas):  # Mostrar episodios con política óptima
    obs, _ = rendering_env.reset()
    estado = obs
    final = False
    while not final:
        accion = np.argmax(q_table[estado])
        nuevo_obs, _, terminated, truncated, _ = rendering_env.step(accion)
        final = terminated or truncated
        estado = nuevo_obs
        time.sleep(0.2)  # ralentizar la visualización


# Mapa de flechas para representar la política óptima
flechas = ['←', '↓', '→', '↑']

print("\nPolítica final con mapa del entorno:\n")
for fila in range(4):
    for col in range(4):
        tile = desc[fila][col]
        if tile in "H":  # Hoyo
            print("H", end=' ')
        elif tile in "G":  # Meta
            print("G", end=' ')
        else:
            estado = fila * 4 + col
            mejor_accion = np.argmax(q_table[estado])
            print(flechas[mejor_accion], end=' ')
    print()

np.set_printoptions(precision=2, suppress=True)
print(q_table)

training_env.close()
rendering_env.close()