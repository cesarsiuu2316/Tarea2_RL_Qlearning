import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
ENV_NAME = 'FrozenLake-v1'

# Parámetros globales
MAX_STEPS = 90
EVAL_EPISODES = 100
TRAIN_EPISODES = 20000


def create_env(slippery=True, render_mode=None):
    env_base = gym.make(ENV_NAME, desc=desc, is_slippery=slippery, render_mode=render_mode)
    return env_base


def train_random_agent():
    env = create_env()
    listado_recompensas = []
    print("RESUMEN DE ENTRENAMIENTO CON RANDOM POLICY")

    for generacion in range(TRAIN_EPISODES):
        estado, _ = env.reset()
        recompensa_total = 0
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = env.action_space.sample()
            estado, recompensa, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            recompensa_total += recompensa
            steps += 1
        listado_recompensas.append(recompensa_total)

        # Mostrar info cada EVAL_EPISODES
        if (generacion + 1) % EVAL_EPISODES == 0: 
            promedio = np.mean(listado_recompensas[-100:])
            print(f'Generación {generacion+1} - Promedio últimas 100 recompensas: {promedio:.4f}')
        
        env.close()

    return listado_recompensas


def train_q_learning_agent(): 
    # Parámetros del entorno
    env = create_env()
    num_estados = env.observation_space.n # 16
    num_acciones = env.action_space.n # 4
    # Al inicio la matriz se debe inicializar en 0 con num_estados filas y num_acciones columnas
    q_table = np.zeros((num_estados, num_acciones)) 

    tasa_aprendizaje = 0.15 # ALPHA
    factor_descuento = 0.99 # GAMMA
    # Epsilon-greedy
    epsilon = 1.0 # Exploración inicial
    epsilon_min = 0.05 # Epsilon mínimo
    decaimiento = 0.9995  # Factor de decaimiento de epsilon
    # Otras variables
    listado_recompensas = [] # Almacenar recompensas por generación
    print("RESUMEN DE ENTRENAMIENTO CON APRENDIZAJE Q-LEARNING CON EPSILON-GREEDY POLICY ")

    for generacion in range(TRAIN_EPISODES):
        estado, _ = env.reset() # Reiniciar el entorno y obtener el estado inicial
        final = False # Variable para indicar si el episodio ha terminado
        recompensa_total = 0 # Recompensa total del episodio
        steps = 0 # Contador de pasos

        while not final and steps < MAX_STEPS: # Mientras no se alcance un estado terminal y no se exceda el máximo de pasos
            if np.random.rand() > epsilon: # Explorar o explotar
                accion = np.argmax(q_table[estado]) # Explotar: elegir la acción con el valor Q más alto
            else:
                accion = env.action_space.sample() #acción aleatoria entre 0 y 3

            nuevo_estado, recompensa, terminated, truncated, _ = env.step(accion) # Realizar la acción en el entorno
            final = terminated or truncated # El episodio ha terminado si se ha alcanzado un estado terminal o truncado

            q_table[estado][accion] += tasa_aprendizaje * (
                recompensa + factor_descuento * np.max(q_table[nuevo_estado]) - q_table[estado][accion]
                ) # Actualizar la Q-table con fórmula de Q-learning: Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))

            estado = nuevo_estado
            recompensa_total += recompensa
            steps += 1 # Incrementar el contador de pasos

        listado_recompensas.append(recompensa_total)
        epsilon = max(epsilon_min, epsilon * decaimiento) # Actualizar epsilon para la siguiente iteración
                
        # Mostrar info cada EVAL_EPISODES
        if (generacion + 1) % EVAL_EPISODES == 0: 
            promedio = np.mean(listado_recompensas[-100:])
            print(f'Generación {generacion+1} - Promedio últimas 100 recompensas: {promedio:.4f} - Épsilon: {epsilon:.4f}')

        env.close()

    return q_table, listado_recompensas


def test_agent(env, policy_fn): # Greedy policy o random policy function
    conteo_exitos = 0
    recompensas = []
    longitudes = []

    for _ in range(EVAL_EPISODES):
        estado, _ = env.reset()
        done = False
        recompensa_total = 0
        steps = 0

        while not done and steps < MAX_STEPS:
            accion = policy_fn(estado)
            estado, recompensa, terminated, truncated, _ = env.step(accion)
            done = terminated or truncated
            recompensa_total += recompensa
            steps += 1

        recompensas.append(recompensa_total) # Almacenar recompensa total del episodio
        longitudes.append(steps) # Almacenar la longitud del episodio
        conteo_exitos += int(recompensa_total > 0) # Contar éxito si la recompensa es positiva, llega a la meta

    recompensa_media = np.mean(recompensas)
    longitud_promedio = np.mean(longitudes)
    porcentaje_exito = (conteo_exitos / EVAL_EPISODES) * 100
    return porcentaje_exito, recompensa_media, longitud_promedio


def test_agent_q_learning_visual(q_table, n_pruebas): # Greedy policy o random policy function
    env = create_env(render_mode='human')
    velocidad = 0.1
    for _ in range(EVAL_EPISODES):
        estado, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < n_pruebas:
            accion = np.argmax(q_table[estado])
            estado, recompensa, terminated, truncated, _ = env.step(accion)
            done = terminated or truncated
            steps += 1
            time.sleep(velocidad)
    env.close()


def render_policy(q_table):
    flechas = ['←', '↓', '→', '↑']
    for fila in range(4):
        for col in range(4):
            tile = desc[fila][col]
            if tile == "H":
                print("H", end=' ')
            elif tile == "G":
                print("G", end=' ')
            else:
                estado = fila * 4 + col
                mejor_accion = np.argmax(q_table[estado])
                print(flechas[mejor_accion], end=' ')
        print()


def promedio_movil(data, block_size=100):
    return np.convolve(data, np.ones(block_size)/block_size, mode='valid')


def plot_recompensas_promedios_moviles(recompensa1, recompensa2, block_size=20):
    avg1 = promedio_movil(recompensa1, block_size)
    avg2 = promedio_movil(recompensa2, block_size)

    plt.figure(figsize=(18, 6))
    plt.plot(avg1, label=f"Random Agent (Media móvil {block_size})")
    plt.plot(avg2, label=f"Q-Learning Agent (Media móvil {block_size})")
    plt.xlabel("Episodio")
    plt.ylabel("Tasa de éxito")
    plt.legend()
    plt.title("Tasa de éxito por episodio (promedio móvil)")
    plt.grid(True)
    plt.show()


def plot_recompensas_crudas(recompensa1, recompensa2):
    plt.figure(figsize=(18, 6))
    plt.plot(recompensa1, label="Random Agent")
    plt.plot(recompensa2, label="Q-Learning Agent")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.title("Recompensas por Episodio")
    plt.show()


def main():
    print("Entrenando agente aleatorio...")
    recompensas_random = train_random_agent()

    print("Entrenando agente Q-Learning...")
    q_table, recompensas_q_learning = train_q_learning_agent()

    # Evaluación
    print("\nEvaluando agentes...")
    env_eval = create_env()
    porcentaje_exito_r, recompensa_media_r, longitud_promedio_r = test_agent(env_eval, lambda estado: env_eval.action_space.sample())
    porcentaje_exito_q, recompensa_media_q, longitud_promedio_q = test_agent(env_eval, lambda estado: np.argmax(q_table[estado]))
    env_eval.close()

    # Resultados
    print("\n--- Resultados ---")
    print(f"Agente Aleatorio: Porcentaje de éxito = {porcentaje_exito_r:.2f}%, recompensa media = {recompensa_media_r:.2f}, longitud promedio = {longitud_promedio_r:.2f}")
    print(f"Agente Q-Learning: Porcentaje de éxito = {porcentaje_exito_q:.2f}%, recompensa media = {recompensa_media_q:.2f}, longitud promedio = {longitud_promedio_q:.2f}")

    # Política final
    print("\nPolítica final del agente Q-Learning:")
    render_policy(q_table)

    # Gráficas de recompensas medias móviles y crudas
    plot_recompensas_crudas(recompensas_random, recompensas_q_learning)
    plot_recompensas_promedios_moviles(recompensas_random, recompensas_q_learning)

    # Visualización de la política aprendida
    test_agent_q_learning_visual(q_table, 5)


if __name__ == "__main__":
    main()