use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::debug;

use vivalaakam_neuro_utils::{Activation, levenshtein};
use vivalaakam_neuro_utils::random::{
    get_random, get_random_position, get_random_range, get_random_weight,
};

use crate::config::Config;
use crate::connection::Connection;
use crate::link::Link;
use crate::network::Network;
use crate::neuron::Neuron;
use crate::neuron_type::NeuronType;
use crate::node::Node;

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Genome {
    connections: Vec<Connection>,
    nodes: Vec<Node>,
    inputs: u32,
    outputs: u32,
}

impl Genome {
    pub fn new(nodes: Vec<Node>, connections: Vec<Connection>) -> Self {
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
        genome.sort_nodes();

        genome
    }

    pub fn generate_genome(
        inputs: usize,
        outputs: usize,
        hidden: Vec<usize>,
        activation: Option<Activation>,
        config: &Config,
    ) -> Self {
        let mut nodes = vec![];
        let mut connections = vec![];

        let mut layer = vec![];
        let mut counter = 1;

        for _ in 0..inputs {
            layer.push(Node::new(
                NeuronType::Input,
                counter,
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
                    NeuronType::Input,
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

            last_layer = layer.clone();
            nodes = [nodes, layer].concat();
        }

        let mut layer = vec![];

        for i in 0..outputs {
            let node = Node::new(
                NeuronType::Output,
                u32::MAX - (outputs - i) as u32,
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

    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(node);
        self.sort_nodes();
    }

    pub fn get_nodes(&self) -> Vec<Node> {
        self.nodes.to_vec()
    }

    pub fn add_connection(&mut self, connection: Connection) {
        self.connections.push(connection);
    }

    pub fn get_connections(&self) -> Vec<Connection> {
        self.connections.to_vec()
    }

    fn sort_nodes(&mut self) {
        debug!(network = json!(self).to_string(), "sort_nodes enter");

        let mut conns: HashMap<u32, Vec<&Connection>> = HashMap::new();
        let mut positions: HashMap<u32, usize> = HashMap::new();

        for node in self.nodes.iter().enumerate() {
            conns.insert(node.1.get_id(), vec![]);
            positions.insert(node.1.get_id(), node.0);
        }

        debug!(
            conns = json!(conns).to_string(),
            positions = json!(positions).to_string(),
            "sort_nodes conns"
        );

        for connection in &self.connections {
            if connection.get_enabled() {
                debug!(to = connection.get_to(), "sort_nodes connection.get_to()");
                if let Some(nodes) = conns.get_mut(&connection.get_to()) {
                    nodes.push(connection);
                }
            }
        }

        let mut inputs = vec![];
        let mut outputs = vec![];
        let mut hidden = VecDeque::new();

        let nodes = self.get_nodes();
        for node in &nodes {
            match node.get_type() {
                NeuronType::Input => {
                    inputs.push(node);
                }
                NeuronType::Hidden => {
                    hidden.push_back(node);
                }
                NeuronType::Output => {
                    outputs.push(node);
                }
                _ => {}
            }
        }

        let mut viewed = HashSet::new();

        inputs.sort_by_key(|a| a.get_position());

        let mut counter = 0;

        for input in &inputs {
            viewed.insert(input.get_id());

            self.nodes[*positions.get(&input.get_id()).unwrap()].set_position(counter);

            counter += 1;
        }

        debug!(hidden = json!(hidden).to_string(), "sort_nodes hidden");

        while !hidden.is_empty() {
            let Some(current) = hidden.pop_front() else {
                continue;
            };

            let Some(connections) = conns.get(&current.get_id()) else {
                continue;
            };

            debug!(
                viewed = json!(viewed).to_string(),
                connections = json!(connections).to_string(),
                "sort_nodes viewed"
            );

            let all_viewed = connections
                .iter()
                .all(|connection| viewed.contains(&connection.get_from()));

            if all_viewed {
                let Some(pos) = positions.get(&current.get_id()) else {
                    continue;
                };
                self.nodes[*pos].set_position(counter);
                viewed.insert(current.get_id());
                counter += 1;
            } else {
                debug!(current = json!(current).to_string(), "sort_nodes hidden");
                hidden.push_back(current);
            }
        }

        debug!(outputs = json!(outputs).to_string(), "sort_nodes outputs");

        outputs.sort_by_key(|a| a.get_position());

        for output in outputs {
            self.nodes[*positions.get(&output.get_id()).unwrap()].set_position(counter);
            viewed.insert(output.get_id());
            counter += 1;
        }
    }

    pub fn get_network(&self) -> Network {
        let mut neurons = vec![];
        let mut conns: HashMap<u32, Vec<&Connection>> = HashMap::new();
        let mut positions: HashMap<u32, u32> = HashMap::new();
        let mut enabled: HashSet<u32> = HashSet::new();

        for node in &self.nodes {
            conns.insert(node.get_id(), vec![]);
            positions.insert(node.get_id(), node.get_position());
            if node.get_enabled() {
                enabled.insert(node.get_id());
            }
        }

        debug!("get_network conns: {conns:?}");

        for connection in &self.connections {
            if connection.get_enabled()
                && enabled.contains(&connection.get_to())
                && enabled.contains(&connection.get_from())
            {
                debug!("get_network connection.get_to(): {}", connection.get_to());
                let nodes = conns.get_mut(&connection.get_to()).unwrap();
                nodes.push(connection);
            }
        }

        let mut nodes = self.get_nodes();

        nodes.sort_by_key(|a| a.get_position());

        for node in &nodes {
            let connections = conns
                .get(&node.get_id())
                .unwrap()
                .iter()
                .map(|conn| {
                    Link::new(
                        *positions.get(&conn.get_from()).unwrap_or(&0),
                        node.get_position(),
                        conn.get_weight(),
                    )
                })
                .collect::<Vec<_>>();

            neurons.push(Neuron::new(
                node.get_type(),
                node.get_bias(),
                node.get_position(),
                node.get_activation(),
                connections,
            ));
        }

        Network::new(neurons)
    }

    pub fn mutate(&self, child: Option<&Genome>, config: &Config) -> Option<Self> {
        let mut genome = Genome::default();
        self.clone_into(&mut genome);

        debug!("mutate enter: {}", json!(genome));

        if let Some(child) = child {
            if get_random() < config.crossover {
                if let Some(g) = genome.mutate_crossover(child) {
                    genome = g;
                    debug!(genome = json!(genome).to_string(), "mutate crossover");
                }
            }
        }

        if get_random() < config.add_node {
            if let Some(g) = genome.mutate_add_node(config) {
                genome = g;
                debug!(genome = json!(genome).to_string(), "mutate add_node");
            }
        }

        if get_random() < config.add_connection {
            if let Some(g) = genome.mutate_add_connection(config) {
                genome = g;
                debug!(genome = json!(genome).to_string(), "mutate add_connection");
            }
        }

        if get_random() < config.connection_enabled {
            if let Some(g) = genome.mutate_connection_enabled() {
                genome = g;
                debug!(
                    genome = json!(genome).to_string(),
                    "mutate connection_enabled"
                );
            }
        }
        if get_random() < config.connection_weight_prob {
            let retry = get_random_range(1, config.connection_weight_iter);

            for i in 0..retry {
                if let Some(g) = genome.mutate_connection_weight(config) {
                    genome = g;
                    debug!(
                        genome = json!(genome).to_string(),
                        "mutate connection_weight ({i})"
                    );
                }
            }
        }

        if get_random() < config.node_enabled {
            if let Some(g) = genome.mutate_node_enabled(config) {
                genome = g;
                debug!(genome = json!(genome).to_string(), "mutate node_enabled");
            }
        }

        if get_random() < config.node_bias_prob {
            if let Some(g) = genome.mutate_node_bias(config) {
                genome = g;
                debug!(genome = json!(genome).to_string(), "mutate node_bias");
            }
        }

        if get_random() < config.node_activation_prob {
            if let Some(g) = genome.mutate_node_activation(config) {
                genome = g;
                debug!(genome = json!(genome).to_string(), "mutate node_activation");
            }
        }

        Some(genome)
    }

    pub fn mutate_add_node(&self, config: &Config) -> Option<Self> {
        if self.nodes.len() >= config.node_max {
            return None;
        }

        let mut genome = Genome::default();
        self.clone_into(&mut genome);

        let conn = get_random_position(self.connections.len());

        let connection = self.connections[conn].clone();

        let activations = Activation::to_vec();
        let activation = get_random_position(activations.len());

        let node = Node::new(
            NeuronType::Hidden,
            get_random_range(self.inputs, u32::MAX - self.outputs),
            get_random_weight(config.node_bias),
            Some(activations[activation]),
            None,
        );
        let from = Connection::new(connection.get_from(), node.get_id(), 1.0);
        genome.connections.push(from);

        let to = Connection::new(node.get_id(), connection.get_to(), connection.get_weight());
        genome.connections.push(to);

        genome.add_node(node);
        genome.connections[conn].set_enabled(false);

        Some(genome)
    }

    pub fn mutate_add_connection(&self, config: &Config) -> Option<Self> {
        if self.connections.len() >= config.connection_max {
            return None;
        }

        let mut genome = Genome::default();
        self.clone_into(&mut genome);

        let mut exists_connections = HashSet::new();

        for connection in &self.connections {
            exists_connections.insert(connection.get_id());
        }

        let mut applicants = vec![];

        for i in 0..self.nodes.len() - 1 {
            let i_node = self.nodes.get(i).unwrap();
            if !i_node.get_enabled() {
                continue;
            }
            for j in i + 1..self.nodes.len() {
                let j_node = self.nodes.get(j).unwrap();
                if !j_node.get_enabled() {
                    continue;
                }

                let conn_id = match (i_node.get_type(), j_node.get_type()) {
                    (NeuronType::Input, NeuronType::Hidden)
                    | (NeuronType::Input, NeuronType::Output)
                    | (NeuronType::Hidden, NeuronType::Output) => {
                        Some(format!("{}:{}", i_node.get_id(), j_node.get_id()))
                    }
                    _ => None,
                };

                if conn_id.is_some() && !exists_connections.contains(&conn_id.unwrap()) {
                    applicants.push((i_node.get_id(), j_node.get_id()));
                }
            }
        }

        if !applicants.is_empty() {
            let conn = get_random_position(applicants.len());

            let applicant = applicants.get(conn).unwrap();

            genome.add_connection(Connection::new(
                applicant.0,
                applicant.1,
                get_random_weight(config.connection_weight),
            ));
        }

        Some(genome)
    }

    pub fn mutate_node_bias(&self, config: &Config) -> Option<Self> {
        let mut genome = Genome::default();
        self.clone_into(&mut genome);

        let applicants = [self.get_hidden_node_ids(), self.get_output_node_ids()].concat();

        let applicant = applicants
            .get(get_random_position(applicants.len()))
            .unwrap();

        let index = genome.get_node_position_by_id(*applicant)?;

        if let Some(node) = genome.nodes.get_mut(index) {
            node.set_bias(node.get_bias() + get_random_weight(config.node_bias_delta));
        }
        Some(genome)
    }

    pub fn mutate_node_activation(&self, _config: &Config) -> Option<Self> {
        let mut genome = Genome::default();
        self.clone_into(&mut genome);

        let applicants = self.get_input_node_ids();

        let applicant = applicants.get(get_random_position(applicants.len()))?;

        let index = genome.get_node_position_by_id(*applicant)?;

        if let Some(node) = genome.nodes.get_mut(index) {
            let activations = Activation::to_vec();
            let activation = get_random_position(activations.len());
            node.set_activation(activations[activation]);
        }

        Some(genome)
    }

    pub fn mutate_node_enabled(&self, _config: &Config) -> Option<Self> {
        let mut genome = Genome::default();
        self.clone_into(&mut genome);

        let applicants = [self.get_hidden_node_ids(), self.get_input_node_ids()].concat();

        let applicant = applicants.get(get_random_position(applicants.len()))?;

        let index = genome.get_node_position_by_id(*applicant)?;

        if let Some(node) = genome.nodes.get_mut(index) {
            node.toggle_enabled();
        }

        Some(genome)
    }

    pub fn mutate_connection_weight(&self, config: &Config) -> Option<Self> {
        let mut genome = Genome::default();
        self.clone_into(&mut genome);
        let mut max_retry = 10;
        let mut index = None;
        while max_retry > 0 && index.is_none() {
            let conn = get_random_position(self.connections.len());
            let connection = self.connections[conn].clone();

            max_retry -= 1;
            if connection.get_enabled() {
                index = Some(conn)
            }
        }

        let index = match index {
            Some(index) => index,
            None => return None,
        };

        if let Some(connection) = genome.connections.get_mut(index) {
            connection.set_weight(
                connection.get_weight() + get_random_weight(config.connection_weight_delta),
            );
        }

        Some(genome)
    }

    pub fn mutate_connection_enabled(&self) -> Option<Self> {
        let mut genome = self.clone();

        let mut appropriate = HashMap::new();

        for node in &self.nodes {
            appropriate.insert(node.get_id(), vec![]);
        }

        for i in 0..self.connections.len() {
            if let Some(app) = appropriate.get_mut(&self.connections[i].get_to()) {
                app.push(i)
            }
        }

        let indices = appropriate
            .values()
            .flat_map(|a| a.to_vec())
            .collect::<Vec<_>>();

        let conn = indices[get_random_position(indices.len())];

        genome.connections[conn].toggle_enabled();

        Some(genome)
    }

    pub fn mutate_crossover(&self, child: &Genome) -> Option<Self> {
        let mut nodes = self.get_nodes();
        let mut connections = self.get_connections();

        let exists_nodes: HashSet<u32> = HashSet::from_iter(self.get_hidden_node_ids());

        for node in child.get_nodes() {
            if node.get_type() == NeuronType::Hidden && !exists_nodes.contains(&node.get_id()) {
                nodes.push(node.clone());
            }
        }

        let mut exists_connections = HashSet::new();

        for connection in &self.get_connections() {
            exists_connections.insert(connection.get_id());
        }

        for connection in &child.get_connections() {
            if !exists_connections.contains(&connection.get_id()) {
                connections.push(connection.clone());
            }
        }

        debug!("mutate_crossover: nodes {nodes:?}");
        debug!("mutate_crossover: connections {connections:?}");

        Some(Genome::new(nodes, connections))
    }

    pub fn get_distance(&self, child: &Genome) -> i32 {
        let mut parent_nodes = self.get_hidden_node_ids();

        let mut child_nodes = child.get_hidden_node_ids();

        parent_nodes.sort();
        child_nodes.sort();

        levenshtein(parent_nodes, child_nodes).unwrap_or(i32::MAX)
    }

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

    pub fn from_weights(weights: impl IntoIterator<Item=f32>) -> Self {
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
