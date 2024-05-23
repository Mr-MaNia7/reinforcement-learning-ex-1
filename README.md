# Reinforcement Learning Exercises

This repository contains implementations of various reinforcement learning algorithms for two distinct types of problems: the Grid World problem via the OpenAI Gym's `FrozenLake-v1` environment and the Single-State Multi-Armed Bandit problem.

## Description

1. **Grid World Problem**:

   - The implementation includes algorithms such as Value Iteration, Policy Iteration, Q-Learning, Epsilon-Greedy, and Upper Confidence Bound (UCB). The `FrozenLake-v1` environment from OpenAI Gym is used as the Grid World scenario.

2. **Multi-Armed Bandit Problem**:
   - This problem simulates a scenario with a single state but multiple actions (arms) that an agent can take. The implemented algorithms include Epsilon-Greedy, UCB, Thompson Sampling, Value Iteration, and Policy Iteration.

## Installation

To run these scripts, you will need Python and the following packages:

- `numpy`
- `gymnasium` (formerly `gym`)

You can install these packages using pip by running the following command:

```bash
pip install -r requirements.txt
```

## Running the Scripts

- To run the Grid World problem script:

  ```bash
  python3 grid_world_problem.py
  ```

- To run the Multi-Armed Bandit problem script:
  ```bash
  python3 bandit_problem.py
  ```

Each script will output the results of the reinforcement learning algorithms, including the total rewards accumulated and the policies learned by the agents.

## Project Structure

- `grid_world_problem.py`: Contains the implementation of the Grid World problem.
- `bandit_problem.py`: Contains the implementation of the Multi-Armed Bandit problem.

## Contributing

Contributions to improve the algorithms or implementations are welcome. Please fork the repository and submit a pull request with your enhancements.

## License

This project is open-sourced under the MIT License. See the `LICENSE` file for more details.

## Contact

For questions or feedback, please reach out to [abdulkarimgmohammed@gmail.com](mailto:abdulkarimgmohammed@gmail.com).
