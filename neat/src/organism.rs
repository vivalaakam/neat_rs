use std::cmp::Ordering;
use std::sync::Mutex;

use ndarray::Array2;

use vivalaakam_neuro_shared::{OrganismTrait, OrganismTraitError};

use crate::genome::GenomeError;
use crate::network::Network;
use crate::{Config, Genome, NeuronType};

/// Represents an individual in the population, encapsulating a genome and its network.
#[derive(Default)]
pub struct Organism {
    pub genome: Genome,
    pub network: Network,
    fitness: Mutex<f32>,
    stagnation: Mutex<usize>,
    genotype: Vec<u32>,
    id: Option<String>,
}

impl From<GenomeError> for OrganismTraitError {
    fn from(error: GenomeError) -> Self {
        OrganismTraitError::new(Box::new(error))
    }
}

impl Organism {
    /// Creates a new organism from a genome.
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
            stagnation: Mutex::new(0),
            genotype,
            id: None,
        }
    }

    /// Returns the genotype (hidden node IDs) of the organism.
    pub fn get_genotype(&self) -> Vec<u32> {
        self.genotype.to_vec()
    }

    /// Serializes the organism's genome to JSON.
    pub fn as_json(&self) -> String {
        self.genome.as_json()
    }

    /// Sets the organism's unique identifier.
    pub fn set_id(&mut self, id: String) {
        self.id = Some(id)
    }

    /// Returns the organism's unique identifier, if set.
    pub fn get_id(&self) -> Option<&String> {
        self.id.as_ref()
    }
}

impl OrganismTrait<Config> for Organism {
    fn activate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.network.activate(inputs)
    }

    fn activate_matrix(&self, matrix: &Array2<f32>) -> Array2<f32> {
        self.network.activate_matrix(matrix)
    }

    fn set_fitness(&self, fitness: f32) {
        let mut data = self.fitness.lock().unwrap();
        *data = fitness
    }

    fn get_fitness(&self) -> f32 {
        *self.fitness.lock().unwrap()
    }

    fn inc_stagnation(&self) {
        let mut data = self.stagnation.lock().unwrap();
        *data += 1;
    }

    fn get_stagnation(&self) -> usize {
        self.stagnation.lock().unwrap().to_owned()
    }

    fn mutate(&self, child: Option<&Self>, config: &Config) -> Result<Self, OrganismTraitError> {
        let genome = child.map(|organism| &organism.genome);

        self.genome
            .mutate(genome, config)
            .map(Organism::new)
            .map_err(OrganismTraitError::from)
    }
}

impl Clone for Organism {
    fn clone(&self) -> Self {
        Organism {
            genome: self.genome.clone(),
            network: self.network.clone(),
            fitness: Mutex::new(self.get_fitness()),
            stagnation: Mutex::new(self.get_stagnation()),
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
