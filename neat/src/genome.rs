use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;
use tracing::debug;

use vivalaakam_neuro_utils::random::{
    get_random, get_random_position, get_random_range, get_random_weight,
};
use vivalaakam_neuro_utils::{levenshtein, Activation};

use crate::config::Config;
use crate::connection::Connection;
use crate::link::Link;
use crate::network::Network;
use crate::neuron::Neuron;
use crate::neuron_type::NeuronType;
use crate::node::Node;

/// Represents a neural network genome: nodes, connections, and input/output counts.
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Genome {
    connections: Vec<Connection>,
    nodes: Vec<Node>,
    inputs: u32,
    outputs: u32,
}

#[derive(Debug, Error)]
pub enum GenomeError {
    #[error("Max nodes reached")]
    MaxNodes,
    #[error("Max connections reached")]
    MaxConnections,
    #[error("Stacked nodes")]
    SortNodesStacked,
    #[error("Node id not found")]
    AddNodeNodeIdNotFound,
    #[error("Connection weight index not found")]
    ConnectionWeightIndexNotFound,
    #[error("Node bias position not found")]
    NodeBiasPositionNotFound,
    #[error("Node bias applicant not found")]
    NodeBiasApplicantNotFound,
    #[error("Node activation position not found")]
    NodeActivationPositionNotFound,
    #[error("Node activation applicant not found")]
    NodeActivationApplicantNotFound,
    #[error("Node enabled position not found")]
    NodeEnabledPositionNotFound,
    #[error("Node enabled applicant not found")]
    NodeEnabledApplicantNotFound,
}

impl Genome {
    /// Creates a genome from given nodes and connections.
    pub fn new(nodes: Vec<Node>, connections: Vec<Connection>) -> Result<Self, GenomeError> {
        let (inputs, outputs) = nodes.iter().fold((0, 0), |a, b| match b.get_type() {
            NeuronType::Input => (a.0 + 1, a.1),
            NeuronType::Output => (a.0, a.1 + 1),
            _ => a,
        });

        let mut genome = Genome {
            nodes,
            connections,
            inputs,
            outputs,
        };
        genome.sort_nodes()?;

        Ok(genome)
    }

    /// Generates a random genome with specified parameters.
    pub fn generate_genome(
        inputs: usize,
        outputs: usize,
        hidden: Vec<usize>,
        activation: Option<Activation>,
        config: &Config,
    ) -> Result<Self, GenomeError> {
        let mut nodes = vec![];
        let mut connections = vec![];

        let mut layer = vec![];
        let mut counter = 1;

        for i in 0..inputs {
            layer.push(Node::new(
                NeuronType::Input,
                i as u32,
                0f32,
                None,
                Some(counter),
            ));
            counter += 1;
        }

        let mut last_layer = layer.clone();
        nodes = [nodes, layer].concat();

        for l in hidden {
            let mut layer = vec![];
            for _ in 0..l {
                let node = Node::new(
                    NeuronType::Hidden,
                    get_random_range((inputs + outputs) as u32, u32::MAX),
                    get_random_weight(config.node_bias),
                    activation,
                    Some(counter),
                );

                for last in &last_layer {
                    connections.push(Connection::new(
                        last.get_id(),
                        node.get_id(),
                        get_random_weight(config.connection_weight),
                    ));
                }

                layer.push(node);
                counter += 1;
            }

            last_layer.clone_from(&layer);
            nodes = [nodes, layer].concat();
        }

        let mut layer = vec![];

        for i in 0..outputs {
            let node = Node::new(
                NeuronType::Output,
                (config.node_max - outputs + i) as u32,
                get_random_weight(config.node_bias),
                activation,
                Some(counter),
            );

            for last in &last_layer {
                connections.push(Connection::new(
                    last.get_id(),
                    node.get_id(),
                    get_random_weight(config.connection_weight),
                ));
            }

            layer.push(node);
            counter += 1;
        }

        nodes = [nodes, layer].concat();

        Genome::new(nodes, connections)
    }

    /// Adds a node to the genome.
    pub fn add_node(&mut self, node: Node) -> Result<(), GenomeError> {
        self.nodes.push(node);
        self.sort_nodes()
    }

    /// Returns all nodes in the genome.
    pub fn get_nodes(&self) -> Vec<Node> {
        self.nodes.to_vec()
    }

    /// Adds a connection to the genome.
    pub fn add_connection(&mut self, connection: Connection) {
        self.connections.push(connection);
    }

    /// Returns all connections in the genome.
    pub fn get_connections(&self) -> Vec<Connection> {
        self.connections.to_vec()
    }

    fn sort_nodes(&mut self) -> Result<(), GenomeError> {
        let mut incoming: HashMap<u32, Vec<u32>> = HashMap::with_capacity(self.nodes.len());

        for node in &self.nodes {
            incoming.insert(node.get_id(), Vec::new());
        }

        for connection in &self.connections {
            if connection.get_enabled() {
                if let Some(froms) = incoming.get_mut(&connection.get_to()) {
                    froms.push(connection.get_from());
                }
            }
        }

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut hidden = VecDeque::new();

        for (index, node) in self.nodes.iter().enumerate() {
            match node.get_type() {
                NeuronType::Input => inputs.push(index),
                NeuronType::Hidden => hidden.push_back(index),
                NeuronType::Output => outputs.push(index),
                NeuronType::Unknown => {}
            }
        }

        inputs.sort_by_key(|&index| self.nodes[index].get_position());

        let mut viewed = HashSet::with_capacity(self.nodes.len());
        let mut counter = 0;

        for index in inputs {
            viewed.insert(self.nodes[index].get_id());
            self.nodes[index].set_position(counter);
            counter += 1;
        }

        let mut iterations = 0;

        while let Some(current) = hidden.pop_front() {
            let id = self.nodes[current].get_id();
            let Some(froms) = incoming.get(&id) else {
                continue;
            };

            iterations += 1;

            if iterations > 1000 {
                return Err(GenomeError::SortNodesStacked);
            }

            if froms.iter().all(|from| viewed.contains(from)) {
                self.nodes[current].set_position(counter);
                viewed.insert(id);
                counter += 1;
            } else {
                hidden.push_back(current);
            }
        }

        outputs.sort_by_key(|&index| self.nodes[index].get_position());

        for index in outputs {
            self.nodes[index].set_position(counter);
            viewed.insert(self.nodes[index].get_id());
            counter += 1;
        }

        Ok(())
    }

    pub fn get_network(&self) -> Network {
        let mut position_of: HashMap<u32, u32> = HashMap::with_capacity(self.nodes.len());
        let mut enabled: HashSet<u32> = HashSet::with_capacity(self.nodes.len());

        for node in &self.nodes {
            position_of.insert(node.get_id(), node.get_position());
            if node.get_enabled() {
                enabled.insert(node.get_id());
            }
        }

        let mut incoming: HashMap<u32, Vec<Link>> = HashMap::with_capacity(self.nodes.len());

        for connection in &self.connections {
            if connection.get_enabled()
                && enabled.contains(&connection.get_to())
                && enabled.contains(&connection.get_from())
            {
                let from = position_of
                    .get(&connection.get_from())
                    .copied()
                    .unwrap_or(0);
                incoming
                    .entry(connection.get_to())
                    .or_default()
                    .push(Link::new(
                        from,
                        position_of.get(&connection.get_to()).copied().unwrap_or(0),
                        connection.get_weight(),
                    ));
            }
        }

        let mut order: Vec<usize> = (0..self.nodes.len()).collect();
        order.sort_by_key(|&index| self.nodes[index].get_position());

        let neurons = order
            .into_iter()
            .map(|index| {
                let node = &self.nodes[index];
                let connections = incoming.remove(&node.get_id()).unwrap_or_default();
                Neuron::from(node).with_connections(connections)
            })
            .collect();

        Network::new(neurons)
    }

    /// Mutates the genome (add node, add connection, weights, etc).
    pub fn mutate(&self, child: Option<&Genome>, config: &Config) -> Result<Self, GenomeError> {
        let mut genome = Genome::default();
        self.clone_into(&mut genome);

        debug!("mutate enter: {}", json!(genome));

        if let Some(child) = child {
            if get_random() < config.crossover {
                let crossed = genome.mutate_crossover(child);
                if let Ok(crossed) = crossed {
                    genome = crossed;
                    debug!(genome = json!(genome).to_string(), "mutate crossover");
                }
            }
        }

        if get_random() < config.add_node {
            let added = genome.mutate_add_node(config);
            if let Ok(added) = added {
                genome = added;
                debug!(genome = json!(genome).to_string(), "mutate add_node");
            }
        }

        if get_random() < config.add_connection {
            let added = genome.mutate_add_connection_in_place(config);
            if added.is_ok() {
                debug!(genome = json!(genome).to_string(), "mutate add_connection");
            }
        }

        if get_random() < config.connection_enabled {
            let toggled = genome.mutate_connection_enabled_in_place();
            if toggled.is_ok() {
                debug!(
                    genome = json!(genome).to_string(),
                    "mutate connection_enabled"
                );
            }
        }
        if get_random() < config.connection_weight_prob {
            let retry = get_random_range(1, config.connection_weight_iter);

            for i in 0..retry {
                let updated = genome.mutate_connection_weight_in_place(config);
                if updated.is_ok() {
                    debug!(
                        genome = json!(genome).to_string(),
                        "mutate connection_weight ({i})"
                    );
                }
            }
        }

        if get_random() < config.node_enabled {
            let toggled = genome.mutate_node_enabled_in_place();
            if toggled.is_ok() {
                debug!(genome = json!(genome).to_string(), "mutate node_enabled");
            }
        }

        if get_random() < config.node_bias_prob {
            let updated = genome.mutate_node_bias_in_place(config);
            if updated.is_ok() {
                debug!(genome = json!(genome).to_string(), "mutate node_bias");
            }
        }

        if get_random() < config.node_activation_prob {
            let updated = genome.mutate_node_activation_in_place();
            if updated.is_ok() {
                debug!(genome = json!(genome).to_string(), "mutate node_activation");
            }
        }

        Ok(genome)
    }

    /// Mutates the genome by adding a new node.
    pub fn mutate_add_node(&self, config: &Config) -> Result<Self, GenomeError> {
        if self.nodes.len() >= config.node_max {
            return Err(GenomeError::MaxNodes);
        }

        let mut genome = Genome::default();
        self.clone_into(&mut genome);

        let conn = get_random_position(self.connections.len());

        let connection = self.connections[conn].clone();

        let activations = Activation::to_vec();
        let activation = get_random_position(activations.len());

        let exists_connections = self
            .nodes
            .iter()
            .map(|node| node.get_id())
            .collect::<HashSet<_>>();

        let mut max_retry = 10;

        let mut node_id = None;

        while max_retry > 0 && node_id.is_none() {
            let node = get_random_range(self.inputs, config.node_max as u32 - self.outputs);
            if !exists_connections.contains(&node) {
                node_id = Some(node);
            }
            max_retry -= 1;
        }

        let Some(node_id) = node_id else {
            return Err(GenomeError::AddNodeNodeIdNotFound);
        };

        let node = Node::new(
            NeuronType::Hidden,
            node_id,
            get_random_weight(config.node_bias),
            Some(activations[activation]),
            None,
        );
        let from = Connection::new(connection.get_from(), node.get_id(), 1.0);
        genome.connections.push(from);

        let to = Connection::new(node.get_id(), connection.get_to(), connection.get_weight());
        genome.connections.push(to);

        genome.add_node(node)?;
        genome.connections[conn].set_enabled(false);

        Ok(genome)
    }

    /// Mutates the genome by adding a new connection.
    pub fn mutate_add_connection(&self, config: &Config) -> Result<Self, GenomeError> {
        let mut genome = self.clone();
        genome.mutate_add_connection_in_place(config)?;
        Ok(genome)
    }

    fn mutate_add_connection_in_place(&mut self, config: &Config) -> Result<(), GenomeError> {
        if self.connections.len() >= config.connection_max {
            return Err(GenomeError::MaxConnections);
        }

        let mut exists_connections = HashSet::with_capacity(self.connections.len());
        for connection in &self.connections {
            exists_connections.insert((connection.get_from(), connection.get_to()));
        }

        let mut applicants = Vec::new();

        if self.nodes.len() > 1 {
            for i in 0..self.nodes.len() - 1 {
                if !self.nodes[i].get_enabled() {
                    continue;
                }
                let i_type = self.nodes[i].get_type();
                let i_id = self.nodes[i].get_id();
                for j in i + 1..self.nodes.len() {
                    if !self.nodes[j].get_enabled() {
                        continue;
                    }
                    let j_type = self.nodes[j].get_type();
                    let j_id = self.nodes[j].get_id();
                    let pair = match (i_type, j_type) {
                        (NeuronType::Input, NeuronType::Hidden)
                        | (NeuronType::Input, NeuronType::Output)
                        | (NeuronType::Hidden, NeuronType::Output) => Some((i_id, j_id)),
                        _ => None,
                    };
                    if let Some(pair) = pair {
                        if !exists_connections.contains(&pair) {
                            applicants.push(pair);
                        }
                    }
                }
            }
        }

        if !applicants.is_empty() {
            let applicant = applicants[get_random_position(applicants.len())];
            self.add_connection(Connection::new(
                applicant.0,
                applicant.1,
                get_random_weight(config.connection_weight),
            ));
        }

        Ok(())
    }

    /// Mutates the bias of a node.
    pub fn mutate_node_bias(&self, config: &Config) -> Result<Self, GenomeError> {
        let mut genome = self.clone();
        genome.mutate_node_bias_in_place(config)?;
        Ok(genome)
    }

    fn mutate_node_bias_in_place(&mut self, config: &Config) -> Result<(), GenomeError> {
        let applicants = [self.get_hidden_node_ids(), self.get_output_node_ids()].concat();

        let Some(&applicant) = applicants.get(get_random_position(applicants.len())) else {
            return Err(GenomeError::NodeBiasApplicantNotFound);
        };

        let Some(index) = self.get_node_position_by_id(applicant) else {
            return Err(GenomeError::NodeBiasPositionNotFound);
        };

        let node = &mut self.nodes[index];
        node.set_bias(node.get_bias() + get_random_weight(config.node_bias_delta));
        Ok(())
    }

    /// Mutates the activation function of a node.
    pub fn mutate_node_activation(&self, _config: &Config) -> Result<Self, GenomeError> {
        let mut genome = self.clone();
        genome.mutate_node_activation_in_place()?;
        Ok(genome)
    }

    fn mutate_node_activation_in_place(&mut self) -> Result<(), GenomeError> {
        let applicants = self.get_input_node_ids();

        let Some(&applicant) = applicants.get(get_random_position(applicants.len())) else {
            return Err(GenomeError::NodeActivationApplicantNotFound);
        };

        let Some(index) = self.get_node_position_by_id(applicant) else {
            return Err(GenomeError::NodeActivationPositionNotFound);
        };

        let activations = Activation::to_vec();
        let activation = activations[get_random_position(activations.len())];
        self.nodes[index].set_activation(activation);
        Ok(())
    }

    /// Toggles the enabled/disabled state of a node.
    pub fn mutate_node_enabled(&self, _config: &Config) -> Result<Self, GenomeError> {
        let mut genome = self.clone();
        genome.mutate_node_enabled_in_place()?;
        Ok(genome)
    }

    fn mutate_node_enabled_in_place(&mut self) -> Result<(), GenomeError> {
        let applicants = [self.get_hidden_node_ids(), self.get_input_node_ids()].concat();

        let Some(&applicant) = applicants.get(get_random_position(applicants.len())) else {
            return Err(GenomeError::NodeEnabledApplicantNotFound);
        };

        let Some(index) = self.get_node_position_by_id(applicant) else {
            return Err(GenomeError::NodeEnabledPositionNotFound);
        };

        self.nodes[index].toggle_enabled();
        Ok(())
    }

    /// Mutates the weight of a connection.
    pub fn mutate_connection_weight(&self, config: &Config) -> Result<Self, GenomeError> {
        let mut genome = self.clone();
        genome.mutate_connection_weight_in_place(config)?;
        Ok(genome)
    }

    fn mutate_connection_weight_in_place(&mut self, config: &Config) -> Result<(), GenomeError> {
        let mut max_retry = 10;
        let mut index = None;
        while max_retry > 0 && index.is_none() {
            let conn = get_random_position(self.connections.len());
            max_retry -= 1;
            if self.connections[conn].get_enabled() {
                index = Some(conn);
            }
        }

        let Some(index) = index else {
            return Err(GenomeError::ConnectionWeightIndexNotFound);
        };

        let connection = &mut self.connections[index];
        connection.set_weight(
            connection.get_weight() + get_random_weight(config.connection_weight_delta),
        );
        Ok(())
    }

    /// Toggles the enabled/disabled state of a connection.
    pub fn mutate_connection_enabled(&self) -> Result<Self, GenomeError> {
        let mut genome = self.clone();
        genome.mutate_connection_enabled_in_place()?;
        Ok(genome)
    }

    fn mutate_connection_enabled_in_place(&mut self) -> Result<(), GenomeError> {
        let mut appropriate: HashMap<u32, Vec<usize>> = HashMap::with_capacity(self.nodes.len());

        for node in &self.nodes {
            appropriate.insert(node.get_id(), Vec::new());
        }

        for (index, connection) in self.connections.iter().enumerate() {
            if let Some(bucket) = appropriate.get_mut(&connection.get_to()) {
                bucket.push(index);
            }
        }

        let indices = appropriate
            .values()
            .flat_map(|bucket| bucket.iter().copied())
            .collect::<Vec<_>>();

        let conn = indices[get_random_position(indices.len())];
        self.connections[conn].toggle_enabled();
        Ok(())
    }

    /// Performs crossover with another genome.
    pub fn mutate_crossover(&self, child: &Genome) -> Result<Self, GenomeError> {
        let mut nodes = self.get_nodes();
        let mut connections = self.get_connections();

        let exists_nodes: HashSet<u32> = HashSet::from_iter(self.get_hidden_node_ids());

        for node in child.get_nodes() {
            if node.get_type() == NeuronType::Hidden && !exists_nodes.contains(&node.get_id()) {
                nodes.push(node.clone());
            }
        }

        let exists_connections: HashSet<String> = HashSet::from_iter(
            self.get_connections()
                .iter()
                .map(|connection| connection.get_id()),
        );

        for connection in &child.get_connections() {
            if !exists_connections.contains(&connection.get_id()) {
                connections.push(connection.clone());
            }
        }

        debug!("mutate_crossover: nodes {nodes:?}");
        debug!("mutate_crossover: connections {connections:?}");

        Genome::new(nodes, connections)
    }

    /// Calculates the Levenshtein distance between hidden nodes of two genomes.
    pub fn get_distance(&self, child: &Genome) -> i32 {
        let mut parent_nodes = self.get_hidden_node_ids();

        let mut child_nodes = child.get_hidden_node_ids();

        parent_nodes.sort();
        child_nodes.sort();

        levenshtein(&parent_nodes, &child_nodes).unwrap_or(i32::MAX)
    }

    /// Serializes the genome to JSON.
    pub fn as_json(&self) -> String {
        json!(self).to_string()
    }

    pub fn get_node_position_by_id(&self, id: u32) -> Option<usize> {
        self.nodes.iter().position(|node| node.get_id() == id)
    }

    pub fn get_connection_position_by_id(&self, id: String) -> Option<usize> {
        self.connections
            .iter()
            .position(|connection| connection.get_id() == id)
    }

    pub fn get_topology(&self) -> Vec<usize> {
        vec![
            1,
            1,
            self.inputs as usize,
            self.outputs as usize,
            self.nodes.len(),
            self.connections.len(),
        ]
    }

    /// Converts the genome to a flat vector of weights.
    pub fn to_weights(&self) -> Vec<f32> {
        self.get_topology()
            .iter()
            .map(|&size| size as f32)
            .chain(self.nodes.iter().flat_map(|node| node.to_weights()))
            .chain(
                self.connections
                    .iter()
                    .flat_map(|connection| connection.to_weights()),
            )
            .collect()
    }

    /// Creates a genome from a flat vector of weights.
    pub fn from_weights(weights: impl IntoIterator<Item = f32>) -> Self {
        let mut weights = weights.into_iter();

        let network_type = weights.next().expect("got no network type") as usize;
        assert_eq!(network_type, 1);

        let network_version = weights.next().expect("got no network version") as usize;
        assert_eq!(network_version, 1);

        let inputs = weights.next().expect("got no inputs") as u32;
        let outputs = weights.next().expect("got no outputs") as u32;

        let nodes_count = weights.next().expect("got no nodes count") as usize;
        let connections_count = weights.next().expect("got no connections count") as usize;

        let nodes = (0..nodes_count)
            .map(|_| Node::from_weights(&mut weights))
            .collect::<Vec<_>>();

        let connections = (0..connections_count)
            .map(|_| Connection::from_weights(&mut weights))
            .collect::<Vec<_>>();

        if weights.next().is_some() {
            panic!("got too many weights");
        }

        Genome {
            nodes,
            connections,
            inputs,
            outputs,
        }
    }

    pub fn get_all_node_ids(&self) -> Vec<u32> {
        self.nodes.iter().map(|node| node.get_id()).collect()
    }

    pub fn get_hidden_node_ids(&self) -> Vec<u32> {
        self.nodes
            .iter()
            .filter_map(|node| {
                if node.get_type() == NeuronType::Hidden {
                    Some(node.get_id())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_input_node_ids(&self) -> Vec<u32> {
        self.nodes
            .iter()
            .filter_map(|node| {
                if node.get_type() == NeuronType::Input {
                    Some(node.get_id())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_output_node_ids(&self) -> Vec<u32> {
        self.nodes
            .iter()
            .filter_map(|node| {
                if node.get_type() == NeuronType::Input {
                    Some(node.get_id())
                } else {
                    None
                }
            })
            .collect()
    }
}

impl From<String> for Genome {
    fn from(genome: String) -> Self {
        serde_json::from_str(genome.as_str()).unwrap()
    }
}

impl From<&str> for Genome {
    fn from(genome: &str) -> Self {
        serde_json::from_str(genome).unwrap()
    }
}
