# vivalaakam_neuro_neat

A library for working with neuroevolutionary networks using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

## Purpose

- Evolving neural network topologies (NEAT)
- Flexible mutation, crossover, and structure configuration
- Genome serialization/deserialization support
- High performance and easy integration

## Main Structures

- **Config** — evolution and mutation parameters
- **Genome** — network genome (nodes, connections, mutation and crossover methods)
- **Node** — network node (neuron)
- **Connection** — connection between neurons
- **NeuronType** — neuron type (Input, Hidden, Output)
- **Organism** — entity encapsulating genome and network
- **Network** — computational network built from genome

## Usage Example

```rust
use vivalaakam_neuro_neat::{Config, Genome, Node, Connection, NeuronType};

let config = Config::default();
let nodes = vec![
    Node::new(NeuronType::Input, 0, 0.0, None, Some(1)),
    Node::new(NeuronType::Output, 1, 0.0, None, Some(2)),
];
let connections = vec![Connection::new(0, 1, 0.5)];
let genome = Genome::new(nodes, connections).unwrap();
let network = genome.get_network();
let output = network.activate(vec![1.0]);
```

## Methods Reference

### Genome
- `new(nodes, connections)` — Create a genome from nodes and connections.
- `generate_genome(inputs, outputs, hidden, activation, config)` — Generate a random genome.
- `mutate(child, config)` — Mutate genome (add node, add connection, weights, etc).
- `mutate_add_node(config)` — Add a new node via mutation.
- `mutate_add_connection(config)` — Add a new connection via mutation.
- `mutate_node_bias(config)` — Mutate node bias.
- `mutate_node_activation(config)` — Mutate node activation function.
- `mutate_node_enabled(config)` — Toggle node enabled/disabled.
- `mutate_connection_weight(config)` — Mutate connection weight.
- `mutate_connection_enabled()` — Toggle connection enabled/disabled.
- `mutate_crossover(child)` — Crossover with another genome.
- `get_network()` — Build a Network from the genome.
- `get_nodes()` / `get_connections()` — Get all nodes/connections.
- `get_distance(child)` — Levenshtein distance between hidden nodes.
- `as_json()` — Serialize genome to JSON.
- `to_weights()` / `from_weights()` — Convert genome to/from flat weights.

### Node
- `new(neuron_type, id, bias, activation, position)` — Create a node.
- `get_id()` / `get_type()` / `get_bias()` / `get_activation()` / `get_enabled()` — Accessors.
- `set_bias(bias)` / `set_activation(activation)` / `toggle_enabled()` — Mutators.
- `get_position()` / `set_position(position)` — Node position in network.
- `to_weights()` / `from_weights()` — Convert node to/from weights.

### Connection
- `new(from, to, weight)` — Create a connection.
- `get_from()` / `get_to()` / `get_weight()` / `get_enabled()` — Accessors.
- `set_weight(weight)` / `set_enabled(enabled)` / `toggle_enabled()` — Mutators.
- `get_id()` — Unique string id for the connection.
- `to_weights()` / `from_weights()` — Convert connection to/from weights.

### Organism
- `new(genome)` — Create an organism from a genome.
- `activate(inputs)` — Run the network on input vector.
- `activate_matrix(matrix)` — Run the network on input matrix (batch).
- `set_fitness(f32)` / `get_fitness()` — Set/get fitness value.
- `inc_stagnation()` / `get_stagnation()` — Increase/get stagnation counter.
- `mutate(child, config)` — Mutate organism (delegates to genome).
- `get_genotype()` — Get genotype (hidden node ids).
- `as_json()` — Serialize genome to JSON.
- `set_id(id)` / `get_id()` — Set/get organism id.

### Network
- `new(neurons)` — Build a network from neurons.
- `activate(inputs)` — Run the network on input vector.
- `activate_matrix(matrix)` — Run the network on input matrix (batch).

### Link
- `new(from_id, to_id, weight)` — Create a link.
- `get_from()` / `get_to()` / `get_weight()` — Accessors.

### NeuronType
- Enum: `Input`, `Hidden`, `Output`, `Unknown`.
- `to_bytes()` / `from_bytes(byte)` — Convert to/from byte.

### Config
- All fields are public. See `src/config.rs` for details and defaults.

## Memory Bank (Quick Reference)

- **Config**: mutation parameters, limits, probabilities (see `src/config.rs`)
- **Genome**:
  - `Genome::new(nodes, connections)` — create a genome
  - `Genome::generate_genome(inputs, outputs, hidden, activation, &config)` — generate a random genome
  - `genome.mutate(...)` — mutate genome
  - `genome.get_network()` — get computational network
  - `genome.as_json()` — serialize to JSON
- **Node**:
  - `Node::new(NeuronType, id, bias, activation, position)`
  - Methods: `get_id()`, `get_type()`, `get_bias()`, `get_activation()`, `get_enabled()`
- **Connection**:
  - `Connection::new(from, to, weight)`
  - Methods: `get_from()`, `get_to()`, `get_weight()`, `get_enabled()`
- **Organism**:
  - `Organism::new(genome)`
  - Methods: `activate(inputs)`, `set_fitness(f32)`, `get_fitness()`

## Tests

See `tests/genome.rs` for usage examples and tests for mutation, crossover, and serialization.

---

**Author:** Andrey Makarov <viva.la.akam@gmail.com>

License: MIT 