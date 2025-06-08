# **Reinforcement Learning Example Using QLearning**

## **Prerequisites**
**Python** must be installed on your system.

## **Steps to Run the Project**

### 1. Clone the repository

```bash
git clone https://github.com/cesarsiuu2316/Tarea2_RL_Qlearning.git
```

### 2. Create and Activate a Virtual Environment
It is recommended to use a virtual environment to install dependencies. Follow these steps:

```bash
# Create the virtual environment using pip
python -m venv venv
```

For uv use the following code:
```bash
# Create the virtual environment using uv
# Install uv if necessary
pip install uv

# Create the virtual environment
uv venv
```

Activate the virtual environment
```bash
# Activate environment
.venv\Scripts\activate
```

### 3. **Install Dependencies**
Once the virtual environment is active, use the `requirements.txt` file from the repository to install the necessary dependencies. Run the following command:

```bash
pip install -r requirements.txt
```
or
```bash
uv add -r requirements.txt
```

### 4. Finally, run the following python programs:

```bash
# Run python projects
python q-learning.py
```
or
```bash
# Run using
uv run q-learning.py
```