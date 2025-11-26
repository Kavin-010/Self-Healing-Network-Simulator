"""
Self-Healing Network using Reinforcement Learning
A network security system that autonomously detects and recovers from cyberattacks
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import random
from collections import defaultdict
import time

class NetworkNode:
    """Represents a node in the network"""
    HEALTHY = 0
    INFECTED = 1
    BLOCKED = 2
    RECOVERING = 3
    
    def __init__(self, node_id, pos):
        self.id = node_id
        self.pos = pos
        self.state = self.HEALTHY
        self.infection_level = 0.0
        self.connections = []
        
    def infect(self, level=0.5):
        if self.state != self.BLOCKED:
            self.state = self.INFECTED
            self.infection_level = min(1.0, self.infection_level + level)
    
    def heal(self):
        self.state = self.RECOVERING
        self.infection_level = max(0, self.infection_level - 0.3)
        if self.infection_level < 0.1:
            self.state = self.HEALTHY
            self.infection_level = 0
    
    def block(self):
        self.state = self.BLOCKED
        
    def get_color(self):
        if self.state == self.HEALTHY:
            return '#00ff88'
        elif self.state == self.INFECTED:
            return f'rgb({int(255 * self.infection_level)}, 0, {int(50 * (1-self.infection_level))})'
        elif self.state == self.BLOCKED:
            return '#888888'
        elif self.state == self.RECOVERING:
            return '#ffaa00'

class SelfHealingNetwork:
    """Main network with Q-Learning agent"""
    
    def __init__(self, num_nodes=15):
        self.num_nodes = num_nodes
        self.nodes = []
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.actions = ['observe', 'block', 'reroute', 'heal']
        
        # Hyperparameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.3  # Exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05
        
        # Statistics
        self.episode = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.healed_count = 0
        self.blocked_attacks = 0
        self.action_history = []
        
        self._initialize_network()
    
    def _initialize_network(self):
        """Create network topology"""
        # Arrange nodes in a circle
        angles = np.linspace(0, 2 * np.pi, self.num_nodes, endpoint=False)
        radius = 3
        
        for i in range(self.num_nodes):
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])
            self.nodes.append(NetworkNode(i, (x, y)))
        
        # Create connections (ring + some random connections)
        for i in range(self.num_nodes):
            # Connect to neighbors
            next_node = (i + 1) % self.num_nodes
            self.nodes[i].connections.append(next_node)
            
            # Add some random connections
            if random.random() > 0.6:
                random_node = random.randint(0, self.num_nodes - 1)
                if random_node != i:
                    self.nodes[i].connections.append(random_node)
    
    def get_state(self):
        """Get current network state"""
        infected = sum(1 for n in self.nodes if n.state == NetworkNode.INFECTED)
        blocked = sum(1 for n in self.nodes if n.state == NetworkNode.BLOCKED)
        healthy = sum(1 for n in self.nodes if n.state == NetworkNode.HEALTHY)
        
        return (infected, blocked, healthy)
    
    def introduce_attack(self):
        """Randomly infect nodes"""
        if random.random() < 0.25:
            healthy_nodes = [n for n in self.nodes if n.state == NetworkNode.HEALTHY]
            if healthy_nodes:
                target = random.choice(healthy_nodes)
                target.infect(random.uniform(0.3, 0.8))
                return target.id
        return None
    
    def spread_infection(self):
        """Spread infection to connected nodes"""
        infected_nodes = [n for n in self.nodes if n.state == NetworkNode.INFECTED]
        
        for node in infected_nodes:
            if random.random() < 0.3 * node.infection_level:
                for conn_id in node.connections:
                    conn_node = self.nodes[conn_id]
                    if conn_node.state == NetworkNode.HEALTHY:
                        conn_node.infect(node.infection_level * 0.5)
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            if not q_values:
                return random.choice(self.actions)
            return max(q_values, key=q_values.get)
    
    def execute_action(self, action):
        """Execute the chosen action and return reward"""
        reward = 0
        infected_nodes = [n for n in self.nodes if n.state == NetworkNode.INFECTED]
        
        if not infected_nodes:
            return 0
        
        target = random.choice(infected_nodes)
        
        if action == 'observe':
            reward = -0.1  # Small penalty for inaction
            
        elif action == 'block':
            target.block()
            reward = 5 if target.infection_level > 0.5 else 2
            self.blocked_attacks += 1
            
        elif action == 'reroute':
            # Reroute connections away from infected node
            for node in self.nodes:
                if target.id in node.connections and node.state == NetworkNode.HEALTHY:
                    reward += 3
            reward = max(1, reward)
            
        elif action == 'heal':
            target.heal()
            if target.state == NetworkNode.HEALTHY:
                reward = 10
                self.healed_count += 1
            else:
                reward = 3
        
        # Penalty for infected nodes
        infected_count = sum(1 for n in self.nodes if n.state == NetworkNode.INFECTED)
        reward -= infected_count * 0.5
        
        return reward
    
    def update_q_table(self, state, action, reward, next_state):
        """Q-learning update"""
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q
    
    def step(self):
        """Execute one simulation step"""
        # Get current state
        state = self.get_state()
        
        # Introduce new attacks
        attacked_node = self.introduce_attack()
        
        # Spread existing infections
        self.spread_infection()
        
        # Agent chooses and executes action
        action = self.choose_action(state)
        reward = self.execute_action(action)
        
        # Get next state
        next_state = self.get_state()
        
        # Update Q-table
        self.update_q_table(state, action, reward, next_state)
        
        # Update statistics
        self.total_reward += reward
        self.action_history.append(action)
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return reward, action, attacked_node
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episode += 1
        avg_reward = self.total_reward / max(1, len(self.action_history))
        self.episode_rewards.append(avg_reward)
        
        # Reset some nodes
        for node in self.nodes:
            if random.random() < 0.3:
                node.state = NetworkNode.HEALTHY
                node.infection_level = 0
        
        return avg_reward

def visualize_network(network, steps=500):
    """Animate the self-healing network"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Self-Healing Network using Reinforcement Learning', 
                 fontsize=16, fontweight='bold')
    
    # Network visualization
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Network Status', fontsize=12, fontweight='bold')
    
    # Statistics
    ax2.set_xlim(0, steps)
    ax2.set_ylim(-10, 15)
    ax2.set_title('Reward Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.grid(True, alpha=0.3)
    
    # Action distribution
    ax3.set_title('Action Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Action')
    ax3.set_ylabel('Count')
    
    rewards_history = []
    step_count = [0]
    
    # Info text
    info_text = ax1.text(0, -3.5, '', ha='center', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate(frame):
        ax1.clear()
        ax1.set_xlim(-4, 4)
        ax1.set_ylim(-4, 4)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title('Network Status', fontsize=12, fontweight='bold')
        
        # Execute simulation step
        reward, action, attacked = network.step()
        rewards_history.append(reward)
        step_count[0] += 1
        
        # Draw connections
        for node in network.nodes:
            for conn_id in node.connections:
                conn_node = network.nodes[conn_id]
                ax1.plot([node.pos[0], conn_node.pos[0]], 
                        [node.pos[1], conn_node.pos[1]], 
                        'gray', alpha=0.2, linewidth=1)
        
        # Draw nodes
        for node in network.nodes:
            if node.state == NetworkNode.HEALTHY:
                color = '#00ff88'
                size = 300
            elif node.state == NetworkNode.INFECTED:
                color = f'#{int(255*node.infection_level):02x}0033'
                size = 300 + 200 * node.infection_level
            elif node.state == NetworkNode.BLOCKED:
                color = '#666666'
                size = 250
            elif node.state == NetworkNode.RECOVERING:
                color = '#ffaa00'
                size = 300
            
            ax1.scatter(node.pos[0], node.pos[1], c=color, s=size, 
                       edgecolors='white', linewidths=2, zorder=10)
            ax1.text(node.pos[0], node.pos[1], str(node.id), 
                    ha='center', va='center', fontsize=8, 
                    fontweight='bold', color='white', zorder=11)
        
        # Update reward plot
        ax2.clear()
        ax2.set_xlim(0, steps)
        ax2.set_ylim(-10, 15)
        ax2.plot(rewards_history, color='#00aaff', linewidth=2)
        if len(rewards_history) > 20:
            moving_avg = np.convolve(rewards_history, np.ones(20)/20, mode='valid')
            ax2.plot(range(19, len(rewards_history)), moving_avg, 
                    color='#ff6600', linewidth=2, label='Moving Avg')
            ax2.legend()
        ax2.set_title('Reward Over Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Update action distribution
        ax3.clear()
        if network.action_history:
            action_counts = {a: network.action_history.count(a) for a in network.actions}
            colors = ['#00ff88', '#666666', '#ffaa00', '#00aaff']
            ax3.bar(action_counts.keys(), action_counts.values(), color=colors)
        ax3.set_title('Action Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Action')
        ax3.set_ylabel('Count')
        
        # Update info
        infected = sum(1 for n in network.nodes if n.state == NetworkNode.INFECTED)
        healthy = sum(1 for n in network.nodes if n.state == NetworkNode.HEALTHY)
        blocked = sum(1 for n in network.nodes if n.state == NetworkNode.BLOCKED)
        
        info = f'Step: {step_count[0]} | Action: {action.upper()} | Reward: {reward:.1f}\n'
        info += f'Healthy: {healthy} | Infected: {infected} | Blocked: {blocked}\n'
        info += f'Total Healed: {network.healed_count} | Attacks Blocked: {network.blocked_attacks}\n'
        info += f'ε: {network.epsilon:.3f} | Avg Reward: {np.mean(rewards_history[-100:]):.2f}'
        
        ax1.text(0, -3.5, info, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        # Reset episode periodically
        if step_count[0] % 100 == 0:
            network.reset_episode()
    
    anim = animation.FuncAnimation(fig, animate, frames=steps, 
                                  interval=100, repeat=False)
    plt.tight_layout()
    plt.show()

# Run the simulation
if __name__ == "__main__":
    print("=" * 60)
    print("Self-Healing Network using Reinforcement Learning")
    print("=" * 60)
    print("\nInitializing network...")
    
    network = SelfHealingNetwork(num_nodes=15)
    
    print(f"Network created with {network.num_nodes} nodes")
    print(f"Actions available: {network.actions}")
    print("\nStarting simulation...\n")
    
    visualize_network(network, steps=500)
    
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print(f"Total episodes: {network.episode}")
    print(f"Total nodes healed: {network.healed_count}")
    print(f"Attacks blocked: {network.blocked_attacks}")
    print(f"Final epsilon: {network.epsilon:.3f}")
    print("=" * 60)