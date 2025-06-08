import gymnasium as gym
import numpy as np
from random import randint

N = 2000
desc = ["SFFF", "FHFH", "FFFH", "HFFG"]

# Entornos
#training_env = gym.make('FrozenLake-v1', desc=desc, is_slippery=True)
training_env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False)
rendering_env = gym.make('FrozenLake-v1', render_mode='human')

# Parámetros del entorno
num_estados = training_env.observation_space.n # 16
num_acciones = training_env.action_space.n # 4

# Al inicio la matriz se debe inicializar en 0
q_table = np.zeros((num_estados, num_acciones))

tasa_aprendizaje = 1 # ALPHA
factor_descuento = 0.99 # GAMMA
generaciones = 100000 # EPISODIOS
listado_recompensas = []

# Epsilon-greedy
epsilon = 1.0
epsilon_min = 0.01
decaimiento = 0.995  # Exponencial

for generacion in range(generaciones):
    observation, _ = training_env.reset()
    estado = observation
    final = False
    recompensa_total = 0

    while not final:
        if np.random.rand() > epsilon:
            accion = np.argmax(q_table[estado])
        else:
            accion = training_env.action_space.sample() #acción aleatoria entre 0 y 3

        nuevo_estado, recompensa, terminated, truncated, _ = training_env.step(accion)
        final = terminated or truncated

        q_table[estado][accion] += tasa_aprendizaje * (
            recompensa + factor_descuento * np.max(q_table[nuevo_estado]) - q_table[estado][accion]
        )

        estado = nuevo_estado
        recompensa_total += recompensa

    listado_recompensas.append(recompensa_total)

    epsilon = max(epsilon_min, epsilon * decaimiento)
    # epsilon = max(epsilon_min, epsilon - decaimiento)

    # Renderizar cada N
    if (generacion + 1) % N == 0:
        obs, _ = rendering_env.reset()
        estado_rend = obs
        final_rend = False
        while not final_rend:
            accion_rend = np.argmax(q_table[estado_rend])
            obs_rend, _, terminated_rend, truncated_rend, _ = rendering_env.step(accion_rend)
            final_rend = terminated_rend or truncated_rend
            estado_rend = obs_rend
            rendering_env.render()
            
    # Mostrar info cada N generaciones
    if (generacion + 1) % N == 0:
        promedio = np.mean(listado_recompensas[-100:])
        print(f'Generación {generacion+1} - Promedio últimas 100 recompensas: {promedio:.4f} - Épsilon: {epsilon:.4f}')

training_env.close()
rendering_env.close()