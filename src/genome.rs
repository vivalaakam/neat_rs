use std::collections::{HashMap, HashSet, VecDeque};

use log::debug;
use new_york_utils::{levenshtein, make_id};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::Config;
use crate::connection::Connection;
use crate::link::Link;
use crate::network::Network;
use crate::neuron::Neuron;
use crate::neuron_type::NeuronType;
use crate::node::Node;
use crate::utils::{get_random, get_random_position, get_random_range, get_random_weight};
use crate::Activation;

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Genome {
    connections: Vec<Connection>,
    nodes: Vec<Node>,
}

impl Genome {
    pub fn new(nodes: Vec<Node>, connections: Vec<Connection>) -> Self {
        let mut genome = Genome { nodes, connections };
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
        for i in 0..inputs {
            layer.push(Node::new(
                NeuronType::Input,
                format!("input_{i}"),
                0f64,
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
                    make_id(6),
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
                format!("output_{i}"),
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
        let mut conns: HashMap<String, Vec<&Connection>> = HashMap::new();
        let mut positions: HashMap<String, usize> = HashMap::new();

        for node in self.nodes.iter().enumerate() {
            conns.insert(node.1.get_id(), vec![]);
            positions.insert(node.1.get_id(), node.0);
        }

        debug!("sort_nodes conns: {:?}", conns);

        for connection in &self.connections {
            if connection.get_enabled() == true {
                debug!("sort_nodes connection.get_to(): {:?}", connection.get_to());
                let nodes = conns.get_mut(&connection.get_to()).unwrap();
                nodes.push(connection);
            }
        }

        let mut inputs = vec![];
        let mut outputs = vec![];
        let mut hidden = VecDeque::new();

        for node in &self.get_nodes() {
            match node.get_type() {
                NeuronType::Input => {
                    inputs.push((node.get_id(), node.get_position()));
                }
                NeuronType::Hidden => {
                    hidden.push_back((node.get_id(), node.get_position()));
                }
                NeuronType::Output => {
                    outputs.push((node.get_id(), node.get_position()));
                }
                _ => {}
            }
        }

        let mut viewed = HashSet::new();

        inputs.sort_by(|a, b| a.1.cmp(&b.1));

        let mut counter = 0;

        for input in &inputs {
            viewed.insert(input.0.to_string());

            self.nodes[*positions.get(&input.0).unwrap()].set_position(counter);

            counter += 1;
        }

        while hidden.len() > 0 {
            let current = hidden.pop_front();

            if current.is_none() {
                continue;
            }

            let current = current.unwrap();

            let connections = conns.get(&current.0).unwrap();

            let all_viewed = connections
                .into_iter()
                .all(|connection| viewed.contains(&connection.get_from()));

            if all_viewed {
                self.nodes[*positions.get(&current.0.to_string()).unwrap()].set_position(counter);
                viewed.insert(current.0.to_string());
                counter += 1;
            } else {
                hidden.push_back(current);
            }
        }

        outputs.sort_by(|a, b| a.1.cmp(&b.1));

        for output in outputs {
            self.nodes[*positions.get(&output.0).unwrap()].set_position(counter);
            viewed.insert(output.0.to_string());
            counter += 1;
        }
    }

    pub fn get_network(&self) -> Network {
        let mut neurons = vec![];
        let mut conns: HashMap<String, Vec<&Connection>> = HashMap::new();
        let mut positions: HashMap<String, usize> = HashMap::new();
        let mut enabled: HashSet<String> = HashSet::new();

        for node in &self.nodes {
            conns.insert(node.get_id(), vec![]);
            positions.insert(node.get_id(), node.get_position());
            if node.get_enabled() {
                enabled.insert(node.get_id());
            }
        }

        debug!("get_network conns: {:?}", conns);

        for connection in &self.connections {
            if connection.get_enabled() == true
                && enabled.contains(&connection.get_to())
                && enabled.contains(&connection.get_from())
            {
                debug!("get_network connection.get_to(): {}", connection.get_to());
                let nodes = conns.get_mut(&connection.get_to()).unwrap();
                nodes.push(connection);
            }
        }

        let mut nodes = self.get_nodes();

        nodes.sort_by(|a, b| a.get_position().cmp(&b.get_position()));

        for node in &nodes {
            let connections = conns.get(&node.get_id()).unwrap();

            neurons.push(Neuron::new(
                node.get_type(),
                node.get_bias(),
                node.get_position(),
                node.get_activation(),
                connections
                    .into_iter()
                    .map(|conn| {
                        Link::new(
                            *positions.get(&conn.get_from()).unwrap_or(&0),
                            node.get_position(),
                            conn.get_weight(),
                        )
                    })
                    .collect::<Vec<_>>(),
            ));
        }

        Network::new(neurons)
    }

    pub fn mutate(&self, child: Option<&Genome>, config: &Config) -> Option<Self> {
        let mut genome = self.clone();
        debug!("mutate enter: {}", json!(genome));

        if child.is_some() && get_random() < config.crossover {
            genome = genome.mutate_crossover(child.unwrap()).unwrap_or(genome);
            debug!("mutate crossover: {}", json!(genome));
        }

        if get_random() < config.add_node {
            genome = genome.mutate_add_node(config).unwrap_or(genome);
            debug!("mutate add_node: {}", json!(genome));
        }

        if get_random() < config.add_connection {
            genome = genome.mutate_add_connection(config).unwrap_or(genome);
            debug!("mutate add_connection: {}", json!(genome));
        }

        if get_random() < config.connection_enabled {
            genome = genome.mutate_connection_enabled().unwrap_or(genome);
            debug!("mutate connection_enabled: {}", json!(genome));
        }
        if get_random() < config.connection_weight_prob {
            let retry = get_random_range(1, config.connection_weight_iter);

            for i in 0..retry {
                genome = genome.mutate_connection_weight(config).unwrap_or(genome);
                debug!("mutate connection_weight ({i}): {}", json!(genome));
            }
        }

        if get_random() < config.node_enabled {
            genome = genome.mutate_node_enabled(config).unwrap_or(genome);
            debug!("mutate node_enabled: {}", json!(genome));
        }

        if get_random() < config.node_bias_prob {
            genome = genome.mutate_node_bias(config).unwrap_or(genome);
            debug!("mutate connection_weight: {}", json!(genome));
        }

        if get_random() < config.node_activation_prob {
            genome = genome.mutate_node_activation(config).unwrap_or(genome);
            debug!("mutate connection_weight: {}", json!(genome));
        }

        Some(genome)
    }

    pub fn mutate_add_node(&self, config: &Config) -> Option<Self> {
        if self.nodes.len() >= config.node_max {
            return None;
        }

        let mut genome = self.clone();

        let conn = get_random_position(self.connections.len());

        let connection = self.connections[conn].clone();

        let activations = Activation::to_vec();
        let activation = get_random_position(activations.len());

        let node = Node::new(
            NeuronType::Hidden,
            make_id(6),
            get_random_weight(config.node_bias),
            Some(activations[activation]),
            None,
        );
        let from = Connection::new(connection.get_from(), node.get_id(), 1f64);
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

        let mut genome = self.clone();
        let mut exists_connections = HashSet::new();

        for connection in &self.connections {
            exists_connections.insert(connection.get_id());
        }

        let mut applicants = vec![];

        for i in 0..self.nodes.len() - 1 {
            let i_node = self.nodes.get(i).unwrap();
            if i_node.get_enabled() != true {
                continue;
            }
            for j in i + 1..self.nodes.len() {
                let j_node = self.nodes.get(j).unwrap();
                if j_node.get_enabled() != true {
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

        if applicants.len() > 0 {
            let conn = get_random_position(applicants.len());

            let applicant = applicants.get(conn).unwrap();

            genome.add_connection(Connection::new(
                applicant.0.to_string(),
                applicant.1.to_string(),
                get_random_weight(config.connection_weight),
            ));
        }

        Some(genome)
    }

    pub fn mutate_node_bias(&self, config: &Config) -> Option<Self> {
        let mut genome = self.clone();

        let applicants = genome
            .nodes
            .iter()
            .filter(|node| node.get_type() != NeuronType::Input)
            .map(|node| node.get_id())
            .collect::<Vec<_>>();
        let applicant = applicants
            .get(get_random_position(applicants.len()))
            .unwrap();
        let index = genome.get_node_position_by_id(applicant.to_string());

        if index.is_none() {
            return None;
        }

        match genome.nodes.get_mut(index.unwrap()) {
            Some(node) => {
                node.set_bias(node.get_bias() + get_random_weight(config.node_bias_delta));
            }
            None => {}
        }

        Some(genome)
    }

    pub fn mutate_node_activation(&self, _config: &Config) -> Option<Self> {
        let mut genome = self.clone();

        let applicants = genome
            .nodes
            .iter()
            .filter(|node| node.get_type() == NeuronType::Input)
            .map(|node| node.get_id())
            .collect::<Vec<_>>();
        let applicant = applicants
            .get(get_random_position(applicants.len()))
            .unwrap();
        let index = genome.get_node_position_by_id(applicant.to_string());

        if index.is_none() {
            return None;
        }

        match genome.nodes.get_mut(index.unwrap()) {
            Some(node) => {
                let activations = Activation::to_vec();
                let activation = get_random_position(activations.len());
                node.set_activation(activations[activation]);
            }
            None => {}
        }

        Some(genome)
    }

    pub fn mutate_node_enabled(&self, _config: &Config) -> Option<Self> {
        let mut genome = self.clone();

        let applicants = genome
            .nodes
            .iter()
            .filter(|node| node.get_type() != NeuronType::Input)
            .map(|node| node.get_id())
            .collect::<Vec<_>>();
        let applicant = applicants
            .get(get_random_position(applicants.len()))
            .unwrap();
        let index = genome.get_node_position_by_id(applicant.to_string());

        if index.is_none() {
            return None;
        }

        match genome.nodes.get_mut(index.unwrap()) {
            Some(node) => {
                node.toggle_enabled();
            }
            None => {}
        }

        Some(genome)
    }

    pub fn mutate_connection_weight(&self, config: &Config) -> Option<Self> {
        let mut genome = self.clone();
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

        if index.is_none() {
            return None;
        }

        match genome.connections.get_mut(index.unwrap()) {
            Some(connection) => {
                connection.set_weight(
                    connection.get_weight() + get_random_weight(config.connection_weight_delta),
                );
            }
            None => {}
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
            match appropriate.get_mut(&self.connections[i].get_to()) {
                Some(app) => app.push(i),
                None => {}
            }
        }

        let indicies = appropriate
            .values()
            .into_iter()
            .flat_map(|a| a.to_vec())
            .collect::<Vec<usize>>();

        let conn = indicies[get_random_position(indicies.len())];

        genome.connections[conn].toggle_enabled();

        Some(genome)
    }

    pub fn mutate_crossover(&self, child: &Genome) -> Option<Self> {
        let mut nodes = self.get_nodes();
        let mut connections = self.get_connections();

        let mut exists_nodes = HashSet::new();

        for node in &self.get_nodes() {
            if node.get_type() == NeuronType::Hidden {
                exists_nodes.insert(node.get_id());
            }
        }

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

        debug!("mutate_crossover: nodes {:?}", nodes);
        debug!("mutate_crossover: connections {:?}", connections);

        Some(Genome::new(nodes, connections))
    }

    pub fn get_distance(&self, child: &Genome) -> i32 {
        let mut parent_nodes = self
            .get_nodes()
            .iter()
            .filter(|node| node.get_type() == NeuronType::Hidden)
            .map(|node| node.get_id())
            .collect::<Vec<_>>();

        let mut child_nodes = child
            .get_nodes()
            .iter()
            .filter(|node| node.get_type() == NeuronType::Hidden)
            .map(|node| node.get_id())
            .collect::<Vec<_>>();

        parent_nodes.sort();
        child_nodes.sort();

        match levenshtein(parent_nodes, child_nodes) {
            Ok(val) => val,
            Err(_) => i32::MAX,
        }
    }

    pub fn as_json(&self) -> String {
        json!(self).to_string()
    }

    pub fn get_node_position_by_id(&self, id: String) -> Option<usize> {
        self.nodes.iter().position(|node| node.get_id() == id)
    }

    pub fn get_connection_position_by_id(&self, id: String) -> Option<usize> {
        self.connections
            .iter()
            .position(|connection| connection.get_id() == id)
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
