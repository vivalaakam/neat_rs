use rand::{Rng, RngCore};
use tracing::debug;

use crate::config::Config;
use crate::neural_network::NeuralNetwork;

#[derive(Clone)]
pub struct Agent {
    pub network: NeuralNetwork,
    pub crossover_chance: f32,
    pub mutation_chance: f32,
    pub mutation_coeff: f32,
    pub learning_coeff: f32,
    pub learning_chance: f32,
    pub fitness: f32,
    id: String,
    hash: String,
}

impl Agent {
    pub fn new(
        network: NeuralNetwork,
        crossover_chance: f32,
        mutation_chance: f32,
        mutation_coeff: f32,
        learning_coeff: f32,
        learning_chance: f32,
        id: Option<String>,
    ) -> Self {
        let hash = network.get_hash();
        Self {
            network,
            crossover_chance,
            mutation_chance,
            mutation_coeff,
            learning_coeff,
            learning_chance,
            fitness: 0.0,
            id: id.unwrap_or_else(|| format!("new-york-symbol/{}", hash)),
            hash,
        }
    }

    pub fn new_from_config(network: NeuralNetwork, config: &Config, id: Option<String>) -> Self {
        let hash = network.get_hash();
        Self {
            network,
            crossover_chance: config.crossover_chance,
            mutation_chance: config.mutation_chance,
            mutation_coeff: config.mutation_coeff,
            learning_coeff: config.learning_coeff,
            learning_chance: config.learning_chance,
            fitness: 0.0,
            id: id.unwrap_or_else(|| format!("new-york-symbol/{}", hash)),
            hash,
        }
    }

    pub fn evolve(&self, child: &Agent, rng: &mut dyn RngCore) -> Agent {
        let parent_a = self.network.get_weights();
        let parent_b = child.network.get_weights();

        let mut weights = parent_a
            .into_iter()
            .zip(parent_b)
            .map(|(a, b)| {
                if rng.random_bool(self.crossover_chance as f64) {
                    a
                } else {
                    b
                }
            })
            .collect::<Vec<f32>>();

        for gene in weights.iter_mut() {
            if rng.random_bool(self.mutation_chance as _) {
                let sign = if rng.random_bool(0.5) { -1f32 } else { 1f32 };
                *gene += sign * self.mutation_coeff * rng.random::<f32>();
            }
        }

        let weights = self
            .network
            .get_topology()
            .iter()
            .map(|&size| size as f32)
            .chain(weights)
            .collect::<Vec<f32>>();

        debug!("topology: {:?}", self.network.get_topology());

        let network = NeuralNetwork::from_weights(weights);

        Self::new(
            network,
            self.crossover_chance,
            self.mutation_chance,
            self.mutation_coeff,
            self.learning_coeff,
            self.learning_chance,
            None,
        )
    }

    pub fn learn(&self, rng: &mut dyn RngCore) -> Agent {
        let weights = self
            .network
            .get_weights()
            .iter()
            .map(|w| {
                if rng.random_bool(self.learning_chance as _) {
                    let sign = if rng.random_bool(0.5) { -1f32 } else { 1f32 };
                    w + self.learning_coeff * sign
                } else {
                    *w
                }
            })
            .collect::<Vec<f32>>();

        let weights = self
            .network
            .get_topology()
            .iter()
            .map(|&size| size as f32)
            .chain(weights)
            .collect::<Vec<f32>>();

        debug!("topology: {:?}", self.network.get_topology());

        let network = NeuralNetwork::from_weights(weights);

        Self::new(
            network,
            self.crossover_chance,
            self.mutation_chance,
            self.mutation_coeff,
            self.learning_coeff,
            self.learning_chance,
            None,
        )
    }

    pub fn activate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.network.activate(inputs)
    }

    pub fn set_fitness(&mut self, fitness: f32) {
        self.fitness = fitness;
    }

    pub fn get_hash(&self) -> String {
        self.hash.to_string()
    }

    pub fn get_id(&self) -> String {
        self.id.to_string()
    }

    pub fn get_network(&self) -> &NeuralNetwork {
        &self.network
    }
}
