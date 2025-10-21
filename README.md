# Lightning DQN on the Arcade Environments

The studios environment has all of the pip dependancies, but I also provide a docker container `dockerfiles/fabric_dqn/Dockerfile` and `make run` to load a bash terminal into this container. 

## Quickstart Development Environment

```bash
# Test on CPU, takes about 1min
PYTHONPATH=$PWD:$PYTHONPATH python src/main.py dqn --fabric.accelerator cpu --trainer.total_decisions 100 --agent.no-compile --agent.no-cudagraphs --agent.buffer_size 1000
# Test on an Accelerator, takes about 3min on an T4
PYTHONPATH=$PWD:$PYTHONPATH python src/main.py dqn --fabric.accelerator cuda --trainer.total_decisions 10_000 --agent.buffer_size 10_000
```

## Run DQN and Option DQN

```bash
# Run dqn on CUDA, 40M decisions == 200M frames worth of compute
PYTHONPATH=$PWD:$PYTHONPATH python src/main.py dqn --trainer.total_decisions 40_000_000 --trainer.env_id "ALE/Pong-v5" --trainer.seed 1

# Run an DQN model with access to options, 40M decisions == 200M frames worth of compute
PYTHONPATH=$PWD:$PYTHONPATH python src/main.py option_dqn --trainer.total_decisions 40_000_000 --trainer.env_id "ALE/Pong-v5" --trainer.seed 1
```