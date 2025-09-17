import numpy as np
import random
import json
import os
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

DEBUG_MODE = True

class DQNAgent:
    def __init__(self, state_size, action_size, memory_file="dqn_experiences.json", model_file="dqn_model.h5"):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.memory_file = memory_file
        self.model_file = model_file
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model_file = model_file
        self.memory_file = memory_file
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.load_experiences()
        self.try_load_model()
        
        if DEBUG_MODE:
            print(f"Mode DEBUG: memory_file={memory_file}")
            print(f"Mode DEBUG: model_file={model_file}")
        
    def _build_model(self):
        """Construction du réseau de neurones"""
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(
            loss='huber',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['mae']
        )
        return model

    def try_load_model(self):
        """Tente de charger le modèle de manière robuste"""
        try:
            if os.path.exists(self.model_file):
                print(f"Tentative de chargement du modèle: {self.model_file}")
                self.model = load_model(self.model_file)
                self.target_model = load_model(self.model_file)
                print(f"Modèle chargé avec succès: {self.model_file}")
            else:
                print("Aucun modèle précédent trouvé. Création d'un nouveau modèle.")
        except Exception as e:
            print(f"Impossible de charger le modèle existant: {e}")
            print("Création d'un nouveau modèle...")
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Stocke une expérience dans la mémoire"""
        state = np.array(state).reshape(1, -1) if len(np.array(state).shape) == 1 else state
        next_state = np.array(next_state).reshape(1, -1) if len(np.array(next_state).shape) == 1 else next_state
        
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        self.save_experience(state, action, reward, next_state, done)
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.array(state).reshape(1, -1) if len(np.array(state).shape) == 1 else state
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size=None):
        """Entraîne le modèle sur un batch d'expériences"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return None
            
        minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states, targets = [], []
        
        for state, action, reward, next_state, done in minibatch:
            # S'assurer que state est au format correct
            state = state.reshape(1, -1) if len(state.shape) == 1 else state
            target = self.model.predict(state, verbose=0)
            
            if done:
                target[0][action] = reward
            else:
                # S'assurer que next_state est au format correct
                next_state = next_state.reshape(1, -1) if len(next_state.shape) == 1 else next_state
                next_q_values = self.target_model.predict(next_state, verbose=0)
                max_next_q = np.max(next_q_values[0])
                target[0][action] = reward + self.gamma * max_next_q
            
            states.append(state[0])  # Garder la forme originale pour l'entraînement
            targets.append(target[0])
        
        # Convertir en tableaux numpy avec la bonne forme
        states_array = np.array(states).reshape(-1, self.state_size)
        targets_array = np.array(targets).reshape(-1, self.action_size)
        
        history = self.model.fit(
            states_array, 
            targets_array, 
            epochs=1, 
            verbose=0,
            batch_size=batch_size
        )
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]

    def save_experience(self, state, action, reward, next_state, done):
        """Sauvegarde une expérience dans un fichier JSON"""
        try:
            state_flat = state.flatten().tolist() if hasattr(state, 'flatten') else state
            next_state_flat = next_state.flatten().tolist() if hasattr(next_state, 'flatten') else next_state
            
            exp = {
                "state": state_flat,
                "action": action,
                "reward": reward,
                "next_state": next_state_flat,
                "done": done
            }
            
            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(exp) + "\n")
                
        except Exception as e:
            print(f"Erreur sauvegarde expérience: {e}")

    def load_experiences(self):
        """Charge les expériences depuis le fichier"""
        try:
            if os.path.exists(self.memory_file):
                loaded_count = 0
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            exp = json.loads(line.strip())
                            state = np.array(exp["state"]).reshape(1, -1)
                            next_state = np.array(exp["next_state"]).reshape(1, -1)
                            self.memory.append((state, exp["action"], exp["reward"], next_state, exp["done"]))
                            loaded_count += 1
                        except json.JSONDecodeError:
                            continue
                print(f"{loaded_count} expériences chargées")
        except Exception as e:
            print(f"Erreur chargement expériences: {e}")

    def save_model(self):
        try:
            self.model.save(self.model_file)
            print(f"Modèle sauvegardé: {self.model_file}")
        except Exception as e:
            print(f"Erreur sauvegarde modèle: {e}")

    # def load_model(self):
    #     """Charge le modèle et les poids"""
    #     try:
    #         if os.path.exists(self.model_file):
    #             self.model = load_model(self.model_file)
    #             self.target_model = load_model(self.model_file)
    #             print(f"Modèle chargé: {self.model_file}")
    #     except Exception as e:
    #         print(f"Erreur chargement modèle: {e}")

    def get_training_metrics(self):
        return {
            "memory_size": len(self.memory),
            "epsilon": self.epsilon,
            "exploration_rate": f"{self.epsilon * 100:.1f}%",
            "batch_size": self.batch_size
        }

# Définition des actions disponibles
ACTIONS = [
    "executer_commande",
    "suggérer_pause",
    "proposer_fermeture_onglets",
    "proposer_musique",
    "demander_precisions",
    "aucune_action"
]

dqn_agent = DQNAgent(5, len(ACTIONS))

def get_current_state(last_command_success, time_of_day, cpu_usage, memory_usage, user_mood):
    return np.array([[last_command_success, time_of_day, cpu_usage, memory_usage, user_mood]])

def compute_reward(feedback_type, response_time, command_complexity):
    base_reward = 0
    
    if feedback_type == "positif":
        base_reward = 15
    elif feedback_type == "negatif":
        base_reward = -10
    elif feedback_type == "neutre":
        base_reward = 5
    
    time_penalty = max(0, (response_time - 2) * -2)
    complexity_bonus = command_complexity * 3
    
    return base_reward + time_penalty + complexity_bonus

def analyze_user_sentiment(text):
    positive_words = ["oui", "bon", "super", "génial", "merci", "parfait", "excellent"]
    negative_words = ["non", "mauvais", "nul", "erreur", "bug", "lent", "raté"]
    
    if not text:
        return 0.5
        
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return 0.8
    elif negative_count > positive_count:
        return 0.2
    else:
        return 0.5