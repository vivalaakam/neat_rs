use std::cmp::Ordering;
use std::sync::Mutex;

use ndarray::Array2;

use crate::network::Network;
use crate::{Config, Genome, NeuronType};

#[derive(Default)]
pub struct Organism {
    pub genome: Genome,
    pub network: Network,
    fitness: Mutex<f32>,
    stagnation: usize,
    genotype: Vec<u32>,
    id: Option<String>,
}

impl Organism {
    pub fn new(genome: Genome) -> Self {
        let network = genome.get_network();
        let mut genotype = genome
            .get_nodes()
            .into_iter()
            .filter(|node| node.get_type() == NeuronType::Hidden)
            .map(|node| node.get_id())
            .collect::<Vec<_>>();

        genotype.sort();

        Organism {
            genome,
            network,
            fitness: Mutex::new(0.0),
            stagnation: 0,
            genotype,
            id: None,
        }
    }

    pub fn activate(&self, params: Vec<f32>) -> Vec<f32> {
        self.network.activate(params)
    }

    pub fn activate_matrix(&self, params: &Array2<f32>) -> Array2<f32> {
        self.network.activate_matrix(params)
    }

    pub fn set_fitness(&self, fitness: f32) {
        let mut data = self.fitness.lock().unwrap();
        *data = fitness
    }

    pub fn get_fitness(&self) -> f32 {
        *self.fitness.lock().unwrap()
    }

    pub fn get_genotype(&self) -> Vec<u32> {
        self.genotype.to_vec()
    }

    pub fn mutate(&self, child: Option<&Organism>, config: &Config) -> Option<Self> {
        let genome = child.map(|organism| &organism.genome);

        self.genome.mutate(genome, config).map(Organism::new)
    }

    pub fn as_json(&self) -> String {
        self.genome.as_json()
    }

    pub fn set_stagnation(&mut self, stagnation: usize) {
        self.stagnation = stagnation
    }

    pub fn inc_stagnation(&mut self) {
        self.stagnation += 1;
    }

    pub fn get_stagnation(&mut self) -> usize {
        self.stagnation
    }

    pub fn set_id(&mut self, id: String) {
        self.id = Some(id)
    }

    pub fn get_id(&self) -> Option<&String> {
        self.id.as_ref()
    }
}

impl Clone for Organism {
    fn clone(&self) -> Self {
        Organism {
            genome: self.genome.clone(),
            network: self.network.clone(),
            fitness: Mutex::new(self.get_fitness()),
            stagnation: self.stagnation,
            genotype: self.genotype.clone(),
            id: self.id.clone(),
        }
    }
}

impl From<String> for Organism {
    fn from(data: String) -> Self {
        let genome = serde_json::from_str::<Genome>(data.as_str()).unwrap();
        Organism::new(genome)
    }
}

impl Ord for Organism {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .get_fitness()
            .partial_cmp(&self.get_fitness())
            .unwrap()
    }
}

impl Eq for Organism {}

impl PartialEq for Organism {
    fn eq(&self, other: &Self) -> bool {
        self.get_fitness() == other.get_fitness()
    }
}

impl PartialOrd for Organism {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
