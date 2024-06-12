# WebotsRL: Reinforcement Learning for Path Planning in Webots

## Description

WebotsRL is a project that implements reinforcement learning for path planning in a robotic environment using the Webots simulation platform. The project uses Proximal Policy Optimization (PPO) as the RL model to navigate and plan paths efficiently.

## Installation

### Prerequisites

- Python 3.x
- Webots R2021b or later
- [OpenAI Gym](https://github.com/openai/gym)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

### Setup

1. Clone the repository:
   git clone https://github.com/v2denny/webotsRL.git
   cd webotsRL
2. Create and activate a virtual environment (optional but recommended) using the env.yaml file
3. Install the required packages:
   pip install -r requirements.txt

## Usage

### Running the Simulation

1. Open Webots and load the project world:
   webots worlds/your_world_file.wbt
2. Run the training code:
   python evironmentv6

### Project Structure

- `scripts/`: Contains utils.py and positions.py.
- `worlds/`: Contains Webots world files.
- `FINAL.zip`: The final model.
- `Report.pdf`: Project's report.
- `README.md`: Project documentation.
- `environmentv6.py`: RL environment and model training.


## Acknowledgements

- Webots (https://cyberbotics.com/)
- OpenAI Gym (https://gym.openai.com/)
- Stable Baselines3 (https://stable-baselines3.readthedocs.io/)
















