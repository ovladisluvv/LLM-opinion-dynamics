# LLM-opinion-dynamics
## Modeling opinion dynamics in social networks with LLM-based agents

A research Python framework designed for studying opinion dynamics in social networks within a network of Large Language Model (LLM) agents. This project models how beliefs, biases, and consensus change over time through agent-to-agent interactions, utilizing graph topologies, classical mathematical models of opinion dynamics, and structured prompt engineering

The main goal of the project is to build an experimental stand for comparing formal opinion dynamics models with LLM-based social simulations, while keeping prompts, configurations, model responses, numeric scores, and experiment logs reproducible

## Current status

The project is in active development. The current version contains the basic building blocks for:
* storing agent and neighbor states
* building prompts for participant agents and judge agent
* running a DeGroot baseline model
* converting textual opinions into numeric scores
* logging prompts, responses, scores, and experiment metadata

The next development stage is to connect these components into a complete experiment runner

## Features
The framework is highly customizable and bridges classical mathematical sociology with modern LLM reasoning. Here is what the framework supports:
* **Multi-Agent Simulation**: Initialize agents with initial beliefs, and system prompts to represent diverse demographic groups
  * **LLM-based agents**: each participant is represented as an agent with an initial opinion, current textual opinion, and numeric opinion score
  * **Social influence through neighbors**: each agent receives opinions of neighboring agents together with influence weights
  * **Judge agent**: a separate evaluator converts textual opinions into numeric agreement scores in the range from `0.0` to `1.0`  
* **Classical Opinion Models Integration**:
  * **DeGroot Model Dynamics**: Simulates continuous opinion updating where agents adjust their views based on the weighted influence of their network neighbors
* **Prompt configuration**: participant and judge prompts are stored in a YAML configuration file
* **Track Simulation Status**: 
  * Real-time logging of agent conversations and internal reasoning processes
  * Dynamic tracking of opinion scores on a continuous scale
  * Automatic detection of simulation end-states (e.g., global consensus)
  * Other metadata that is saved into CSV logs
* **Reproducibility-oriented design**:
  * fixed experiment configurations
  * explicit prompt templates
  * separate logging of participant and judge responses
  * support for multiple independent runs
  * planned storage of random seeds, model names, temperatures, and network parameters
 
### Roadmap / Planned Actions
* **Opinion trajectory plots**:
* **Comparison between DeGroot and LLM-agent trajectories**:
* **Aggregation over repeated runs**: 
* **Validation metrics for LLM-based simulations**:
* **Local LLM Support**: Implementation of a wrapper for local inference (vLLM, Ollama) to reduce API costs, ensure privacy, and improve execution speed
* **Dynamic Networks**: Allowing agents to autonomously follow/unfollow others based on opinion similarity or interaction history
* **Interactive Dashboard**: A GUI for real-time visualization of the network graph and opinion trajectories

## Method

The project separates the simulation into two levels:

1. **Textual opinion update**

   A participant agent receives:
   * the main thesis
   * its current opinion
   * its initial stance description
   * opinions of neighboring agents
   * influence weights of the neighbors

   The agent then produces an updated textual opinion.

2. **Numeric opinion scoring**

   A judge agent receives:
   * the thesis
   * the participant's textual opinion

   It returns a numeric score from `0.0` to `1.0`, where:
   * `0.0` means complete disagreement with the thesis
   * `0.5` means neutral, mixed, unclear, or balanced position
   * `1.0` means complete agreement with the thesis

This allows the project to compare natural-language opinion changes with formal numeric trajectories from mathematical models

## Project Structure

The codebase is divided into modules for ease of maintenance and clear separation of tasks:

* **`agents/agent_state.py`** - data structures for participant states, neighbor states, participant results, and judge results
* **`agents/llm_agents.py`** - abstract LLM agent interface, participant agent logic, and judge agent logic
* **`math_models/degroot.py`** - DeGroot opinion dynamics baseline, weight matrix validation, consensus detection, and trajectory storage
* **`prompts/config.yaml`** - YAML configuration file with prompt templates for participant and judge agents
* **`prompts/prompt_builder.py`** - prompt construction utilities for inserting thesis, current opinion, neighbor opinions, and influence weights
* **`utils/logger.py`** - CSV logger for experiment events, prompts, responses, judge outputs, and scores

## Planned architecture

A complete simulation step will follow this pipeline:

1. Load experiment configuration
2. Build or load a social network
3. Initialize agents with textual opinions and numeric scores
4. For each simulation step:
   * collect neighbor states for each agent
   * build participant prompts
   * query participant LLM agents
   * send updated opinions to the judge agent
   * parse numeric opinion scores
   * update agent states
   * log all prompts, responses, and scores
5. Save trajectories and final results
6. Compare LLM-agent dynamics with baseline mathematical models
7. Generate plots and summary tables

## Requirements

To run the project you need:

### Running experiments

- Python 3.11 or newer
- Access to at least one LLM API provider
- API keys configured through environment variables or a local configuration file
- Installed Python dependencies

Planned dependencies:

```bash
numpy
pyyaml
pandas
matplotlib
networkx
