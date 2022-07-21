use std::cmp::Ordering;

use crate::network::Network;
use crate::{Config, Genome};

#[derive(Clone)]
pub struct Organism {
    pub genome: Genome,
    pub fitness: f64,
    pub network: Network,
}

impl Organism {
    pub fn new(genome: Genome) -> Self {
        let network = genome.get_network();
        Organism {
            genome,
            fitness: 0f64,
            network,
        }
    }

    pub fn activate(&self, params: Vec<f64>) -> Vec<f64> {
        self.network.activate(params)
    }

    pub fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }

    pub fn get_fitness(&self) -> f64 {
        self.fitness
    }

    pub fn mutate(&self, child: &Organism, config: &Config) -> Option<Self> {
        match self.genome.mutate(&child.genome, config) {
            None => None,
            Some(genome) => Some(Organism::new(genome)),
        }
    }

    pub fn as_json(&self) -> String {
        self.genome.as_json()
    }

    pub fn from_json(&self, data: String) -> Self {
        let genome = serde_json::from_str::<Genome>(data.as_str()).unwrap();
        Organism::new(genome)
    }
}

impl Ord for Organism {
    fn cmp(&self, other: &Self) -> Ordering {
        other.fitness.partial_cmp(&self.fitness).unwrap()
    }
}

impl Eq for Organism {}

impl PartialEq for Organism {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness
    }
}

impl PartialOrd for Organism {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
