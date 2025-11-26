# Self-Healing-Network-Simulator

This project is a reinforcement learning–based simulation that models how a network can detect, respond to, and recover from cyberattacks. It uses Q-learning to decide actions such as observing the network, blocking infected nodes, healing compromised nodes, and rerouting connections. The simulation also visualizes the network state, infection spread, rewards, and action patterns in real time.

## How it works

Nodes can be Healthy, Infected, Blocked, or Recovering

Infections spread through connected nodes

A Q-learning agent selects actions based on the current network state

Rewards encourage healing, blocking, and reducing infection spread

A live animation shows the agent’s behavior and learning progress

## Future Improvements

Add more realistic network topologies (LAN, data center, cloud-style networks) instead of just a simple ring.

Use more advanced RL methods (like Deep Q-Learning) to handle larger networks and more complex attack patterns.

Integrate real network logs or traffic traces so the simulation is closer to real-world behaviour.

Build a simple dashboard/GUI to control the simulation, visualize attacks, and tune parameters in real time.

Extend the system from simulation to a prototype that can connect to real machines and help detect and recover actual infected systems in a controlled environment.
