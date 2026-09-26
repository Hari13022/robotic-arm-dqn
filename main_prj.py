# main.py
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Optional

# PyBullet imports
import pybullet as p
import pybullet_data
# =========================
# Missing Definitions Fix
# =========================
from collections import namedtuple

# Each transition is one experience tuple (state, action, reward, next_state, done)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Standard Replay Buffer (non-prioritized)
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = Transition(state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)


# ---------------------------
# Enhanced Replay Buffer with Prioritized Experience Replay
# ---------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(Transition(state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = Transition(state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            return []
            
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # small constant to avoid zero

    def __len__(self):
        return self.size

# ---------------------------
# Enhanced DQN with Dueling Architecture
# ---------------------------
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super(DuelingDQN, self).__init__()
        
        # Feature layer
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# ---------------------------
# Enhanced PyBullet Environment
# ---------------------------
class EnhancedKukaEnv:
    def __init__(self, render=True, workspace_limits=None, difficulty="medium"):
        self.render = render
        if render:
            self.physics = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable GUI for cleaner view
        else:
            self.physics = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load environment
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        
        # Get controllable joints
        self.num_joints = p.getNumJoints(self.robot)
        self.joint_indices = [i for i in range(self.num_joints) 
                            if p.getJointInfo(self.robot, i)[2] != p.JOINT_FIXED]
        
        # Joint limits for safety
        self.joint_limits = []
        for ji in self.joint_indices:
            info = p.getJointInfo(self.robot, ji)
            self.joint_limits.append((info[8], info[9]))  # lower, upper limits
        
        # Workspace limits
        self.workspace_limits = workspace_limits or [[-0.8, 0.8], [-0.8, 0.8], [0.1, 0.8]]
        self.difficulty = difficulty
        
        # Create target
        self.target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 1])
        self.target_id = None
        
        # Obstacles for harder difficulty
        self.obstacles = []
        if difficulty == "hard":
            self._create_obstacles()
        
        self.reset()

    def _create_obstacles(self):
        """Create obstacles in the workspace for more challenging tasks"""
        obstacle_color = [0.5, 0.5, 0.5, 1.0]
        # Add some boxes as obstacles
        for i in range(3):
            pos = [0.3 + i*0.2, 0.1, 0.2]
            obstacle = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.1])
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.1], 
                                       rgbaColor=obstacle_color)
            obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obstacle,
                                          baseVisualShapeIndex=visual, basePosition=pos)
            self.obstacles.append(obstacle_id)

    def _get_random_target_position(self):
        """Generate target position based on difficulty"""
        if self.difficulty == "easy":
            return [0.4, 0.0, 0.3 + 0.1 * np.random.randn()]
        elif self.difficulty == "medium":
            return [0.5 + 0.2 * (np.random.rand() - 0.5), 
                   0.2 * (np.random.rand() - 0.5), 
                   0.4 + 0.2 * np.random.rand()]
        else:  # hard
            return [0.6 * (np.random.rand() - 0.5), 
                   0.6 * (np.random.rand() - 0.5), 
                   0.3 + 0.4 * np.random.rand()]

    def reset(self):
        # Reset robot to neutral position
        neutral_positions = [0, 0, 0, np.pi/2, 0, -np.pi/2, 0]  # KUKA iiwa neutral
        for i, ji in enumerate(self.joint_indices):
            if i < len(neutral_positions):
                p.resetJointState(self.robot, ji, targetValue=neutral_positions[i])
        
        # Remove old target and create new one
        if self.target_id is not None:
            p.removeBody(self.target_id)
        
        target_pos = self._get_random_target_position()
        self.target_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                         baseVisualShapeIndex=self.target_visual,
                                         basePosition=target_pos)
        
        # Step simulation to stabilize
        for _ in range(10):
            p.stepSimulation()
            
        return self._get_obs()

    def _get_obs(self):
        # Enhanced observation space
        joint_positions = [p.getJointState(self.robot, i)[0] for i in self.joint_indices]
        joint_velocities = [p.getJointState(self.robot, i)[1] for i in self.joint_indices]
        
        # End-effector information
        ee_state = p.getLinkState(self.robot, self.joint_indices[-1])
        ee_pos = ee_state[0]
        ee_orient = p.getEulerFromQuaternion(ee_state[1])
        
        # Target information
        target_pos = p.getBasePositionAndOrientation(self.target_id)[0]
        
        # Distance to target
        distance_to_target = np.linalg.norm(np.array(ee_pos) - np.array(target_pos))
        
        # Create comprehensive state vector
        state = np.concatenate([
            np.array(joint_positions),
            np.array(joint_velocities),
            np.array(ee_pos),
            np.array(ee_orient),
            np.array(target_pos),
            [distance_to_target]
        ])
        
        return state.astype(np.float32)

    def step(self, action):
        # Enhanced action execution with joint limit safety
        delta = 0.05  # Reduced step size for smoother motion
        
        joint_idx = action // 2
        direction = 1 if action % 2 == 0 else -1
        
        if joint_idx < len(self.joint_indices):
            ji = self.joint_indices[joint_idx]
            current_pos = p.getJointState(self.robot, ji)[0]
            target_pos = current_pos + direction * delta
            
            # Apply joint limits
            lower, upper = self.joint_limits[joint_idx]
            target_pos = np.clip(target_pos, lower, upper)
            
            p.setJointMotorControl2(self.robot, ji, p.POSITION_CONTROL, 
                                  targetPosition=target_pos, force=150)
        
        # Step simulation
        for _ in range(4):  # Multiple substeps for stability
            p.stepSimulation()
            if self.render:
                time.sleep(0.005)
        
        next_state = self._get_obs()
        
        # Enhanced reward calculation
        reward, done, info = self._compute_reward(next_state)
        
        return next_state, reward, done, info

    def _compute_reward(self, state):
        # Extract information from state
        joint_positions = state[:len(self.joint_indices)]
        joint_velocities = state[len(self.joint_indices):2*len(self.joint_indices)]
        ee_pos = state[2*len(self.joint_indices):2*len(self.joint_indices)+3]
        target_pos = state[2*len(self.joint_indices)+6:2*len(self.joint_indices)+9]
        distance = state[-1]
        
        # Base distance reward
        distance_reward = -distance * 2.0
        
        # Success bonus
        success_bonus = 10.0 if distance < 0.02 else 0.0
        
        # Smoothness penalty (encourage smooth motion)
        velocity_penalty = -0.01 * np.sum(np.square(joint_velocities))
        
        # Joint limit penalty
        limit_penalty = 0.0
        for i, (pos, (lower, upper)) in enumerate(zip(joint_positions, self.joint_limits)):
            if pos < lower + 0.1 or pos > upper - 0.1:
                limit_penalty -= 0.1
        
        # Collision check (for hard difficulty)
        collision_penalty = 0.0
        if self.difficulty == "hard":
            for obstacle in self.obstacles:
                contacts = p.getContactPoints(bodyA=self.robot, bodyB=obstacle)
                if contacts:
                    collision_penalty -= 1.0
        
        total_reward = distance_reward + success_bonus + velocity_penalty + limit_penalty + collision_penalty
        
        done = distance < 0.02  # Success condition
        
        info = {
            'distance': float(distance),
            'success': bool(done),
            'joint_limit_violation': limit_penalty < 0,
            'collision': collision_penalty < 0
        }
        
        return float(total_reward), done, info

    def close(self):
        if self.physics >= 0:
            p.disconnect(self.physics)

# ---------------------------
# Enhanced DQN Agent
# ---------------------------
class EnhancedDQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, tau=0.01, device='cpu'):
        self.device = device
        self.model = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        
        self.optim = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.steps = 0
        
        # Loss tracking
        self.loss_history = []

    def act(self, state, eps=0.1):
        if random.random() < eps:
            return random.randrange(self.action_dim)
        
        state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_v)
        return int(q_values.argmax().item())

    def update(self, batch, weights=None):
        if not batch:
            return 0.0
            
        states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32).to(self.device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.int64).to(self.device)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32).to(self.device)
        dones = torch.tensor([float(b.done) for b in batch], dtype=torch.float32).to(self.device)
        
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1]
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        # Compute loss
        if weights is not None:
            loss = (weights * (current_q - target_q) ** 2).mean()
        else:
            loss = nn.functional.mse_loss(current_q, target_q)
        
        # Optimize
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
        self.optim.step()
        
        self.steps += 1
        
        # Soft update target network
        if self.steps % 10 == 0:  # More frequent soft updates
            self.soft_update()
        
        self.loss_history.append(loss.item())
        return loss.item()

    def soft_update(self):
        """Soft update model parameters"""
        for target_param, param in zip(self.target.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'steps': self.steps,
            'loss_history': self.loss_history
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target.load_state_dict(checkpoint['target_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.loss_history = checkpoint['loss_history']

# ---------------------------
# Enhanced Streamlit UI
# ---------------------------
def setup_ui():
    st.set_page_config(layout="wide", page_title="Enhanced DQN-IK Simulation")
    
    st.title("🤖 Enhanced DQN-IK Robotic Simulation")
    st.markdown("""
    This interactive demo trains a DQN agent to control a KUKA robot arm using PyBullet simulation.
    The agent learns to reach target positions while avoiding obstacles and respecting joint limits.
    """)
    
    # Sidebar configuration
    st.sidebar.header("🎯 Simulation Configuration")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        render = st.checkbox("Render GUI", value=True)
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
    with col2:
        use_per = st.checkbox("Use PER", value=True)
        device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    
    st.sidebar.header("⚙️ Training Hyperparameters")
    
    episodes = st.sidebar.slider("Training Episodes", 10, 2000, 500)
    max_steps = st.sidebar.slider("Max Steps per Episode", 50, 500, 200)
    batch_size = st.sidebar.slider("Batch Size", 16, 256, 64)
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        lr = st.number_input("Learning Rate", 1e-5, 1e-2, 1e-3, format="%.4f")
        gamma = st.slider("Discount Factor", 0.8, 0.999, 0.99)
    with col4:
        eps_start = st.slider("Epsilon Start", 0.0, 1.0, 1.0)
        eps_end = st.slider("Epsilon End", 0.0, 0.5, 0.05)
    
    eps_decay = st.sidebar.slider("Epsilon Decay Steps", 100, 10000, 2000)
    
    return {
        'render': render,
        'difficulty': difficulty,
        'use_per': use_per,
        'device': device,
        'episodes': episodes,
        'max_steps': max_steps,
        'batch_size': batch_size,
        'lr': lr,
        'gamma': gamma,
        'eps_start': eps_start,
        'eps_end': eps_end,
        'eps_decay': eps_decay
    }

def create_metrics_plots(metrics):
    """Create interactive plots for training metrics"""
    if not metrics:
        return
    
    df = pd.DataFrame(metrics)
    
    # Create subplots
    fig = go.Figure()
    
    # Reward plot
    fig.add_trace(go.Scatter(x=df['episode'], y=df['reward'], 
                           mode='lines', name='Episode Reward',
                           line=dict(color='blue')))
    
    # Moving average
    window = max(1, len(df) // 20)
    df['reward_ma'] = df['reward'].rolling(window=window, center=True).mean()
    fig.add_trace(go.Scatter(x=df['episode'], y=df['reward_ma'],
                           mode='lines', name=f'Reward MA ({window})',
                           line=dict(color='red', dash='dash')))
    
    fig.update_layout(title='Training Progress', xaxis_title='Episode', yaxis_title='Reward')
    
    return fig

# Main application
def main():
    config = setup_ui()
    
    # Initialize session state
    if 'env' not in st.session_state:
        st.session_state.env = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'buffer' not in st.session_state:
        st.session_state.buffer = PrioritizedReplayBuffer() if config['use_per'] else ReplayBuffer()
    if 'metrics' not in st.session_state:
        st.session_state.metrics = []
    if 'training' not in st.session_state:
        st.session_state.training = False
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        train_btn = st.button("🚀 Start Training", use_container_width=True)
    with col2:
        eval_btn = st.button("📊 Run Evaluation", use_container_width=True)
    with col3:
        stop_btn = st.button("🛑 Stop Training", use_container_width=True)
    with col4:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    
    # Placeholders for dynamic content
    progress_ph = st.empty()
    status_ph = st.empty()
    charts_ph = st.empty()
    metrics_ph = st.empty()
    
    def init_env_agent():
        if st.session_state.env is None:
            st.session_state.env = EnhancedKukaEnv(
                render=config['render'],
                difficulty=config['difficulty']
            )
        
        if st.session_state.agent is None:
            state_dim = len(st.session_state.env._get_obs())
            action_dim = len(st.session_state.env.joint_indices) * 2
            
            st.session_state.agent = EnhancedDQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=config['lr'],
                gamma=config['gamma'],
                device=config['device']
            )
        
        return st.session_state.env, st.session_state.agent
    
    # Training logic
    if train_btn and not st.session_state.training:
        st.session_state.training = True
        env, agent = init_env_agent()
        buffer = st.session_state.buffer
        
        progress_bar = progress_ph.progress(0)
        status_text = status_ph.empty()
        
        global_step = 0
        eps = config['eps_start']
        
        for ep in range(config['episodes']):
            if not st.session_state.training:
                break
                
            state = env.reset()
            ep_reward = 0.0
            ep_success = False
            
            for step in range(config['max_steps']):
                # Epsilon decay
                eps = max(config['eps_end'], 
                         config['eps_start'] - global_step / config['eps_decay'] * 
                         (config['eps_start'] - config['eps_end']))
                
                action = agent.act(state, eps)
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                ep_reward += reward
                global_step += 1
                
                # Training step
                if len(buffer) >= config['batch_size']:
                    if config['use_per']:
                        batch, indices, weights = buffer.sample(config['batch_size'])
                        loss = agent.update(batch, weights)
                        # Update priorities (simplified - in practice, use TD error)
                        buffer.update_priorities(indices, [1.0] * len(indices))
                    else:
                        batch = buffer.sample(config['batch_size'])
                        loss = agent.update(batch)
                
                if done:
                    ep_success = info.get('success', False)
                    break
            
            # Log metrics
            metrics_entry = {
                'episode': ep,
                'reward': ep_reward,
                'steps': step + 1,
                'epsilon': eps,
                'success': ep_success,
                'distance': info.get('distance', 0.0)
            }
            st.session_state.metrics.append(metrics_entry)
            
            # Update progress
            progress = (ep + 1) / config['episodes']
            progress_bar.progress(progress)
            
            status_text.text(
                f"Episode {ep+1}/{config['episodes']} | "
                f"Reward: {ep_reward:.2f} | "
                f"Steps: {step+1} | "
                f"Epsilon: {eps:.3f} | "
                f"Success: {ep_success}"
            )
            
            # Update charts every 10 episodes
            if (ep + 1) % 10 == 0:
                fig = create_metrics_plots(st.session_state.metrics)
                if fig:
                    charts_ph.plotly_chart(fig, use_container_width=True)
        
        # Save model and final metrics
        if st.session_state.agent:
            st.session_state.agent.save("assets/enhanced_dqn_model.pth")
        
        st.session_state.training = False
        status_ph.success("Training completed!")
    
    # Evaluation logic
    if eval_btn:
        env, agent = init_env_agent()
        
        # Load model if available
        model_path = "assets/enhanced_dqn_model.pth"
        if os.path.exists(model_path):
            try:
                agent.load(model_path)
                status_ph.info("Loaded trained model for evaluation")
            except Exception as e:
                status_ph.warning(f"Could not load model: {e}")
        
        eval_episodes = 5
        eval_results = []
        
        for ep in range(eval_episodes):
            state = env.reset()
            ep_reward = 0.0
            steps = 0
            
            for step in range(200):
                action = agent.act(state, eps=0.01)  # Small epsilon for evaluation
                state, reward, done, info = env.step(action)
                ep_reward += reward
                steps += 1
                
                if done:
                    break
            
            eval_results.append({
                'episode': ep + 1,
                'reward': ep_reward,
                'steps': steps,
                'success': info.get('success', False),
                'final_distance': info.get('distance', 0.0)
            })
        
        # Display evaluation results
        eval_df = pd.DataFrame(eval_results)
        metrics_ph.dataframe(eval_df.style.format({
            'reward': '{:.2f}',
            'final_distance': '{:.3f}'
        }))
        
        success_rate = eval_df['success'].mean() * 100
        status_ph.info(f"Evaluation completed! Success rate: {success_rate:.1f}%")
    
    # Stop training
    if stop_btn:
        st.session_state.training = False
        status_ph.warning("Training stopped by user")
    
    # Reset everything
    if reset_btn:
        if st.session_state.env:
            st.session_state.env.close()
        st.session_state.env = None
        st.session_state.agent = None
        st.session_state.buffer = PrioritizedReplayBuffer() if config['use_per'] else ReplayBuffer()
        st.session_state.metrics = []
        st.session_state.training = False
        status_ph.info("Environment reset")
    
    # Display current metrics
    if st.session_state.metrics:
        st.subheader("Training Metrics")
        recent_metrics = pd.DataFrame(st.session_state.metrics[-50:])  # Last 50 episodes
        st.dataframe(recent_metrics.tail(10).style.format({
            'reward': '{:.2f}',
            'epsilon': '{:.3f}',
            'distance': '{:.3f}'
        }))

if __name__ == "__main__":
    main()