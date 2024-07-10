use vivalaakam_neuro_utils::random::get_random_range;

use crate::fitness::FitnessTrait;
use crate::organism::OrganismTrait;

pub struct Population<C, T> {
    size: usize,
    organisms: Vec<T>,
    config: C,
}

impl<C, T> Population<C, T>
where
    C: Clone + Default,
    T: OrganismTrait<C> + std::cmp::Ord + Clone,
{
    pub fn new(size: usize, config: C) -> Self {
        Population {
            size,
            config,
            organisms: vec![],
        }
    }

    pub fn len(&self) -> usize {
        self.organisms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.organisms.is_empty()
    }

    pub fn add_organism(&mut self, organism: T) {
        self.organisms.push(organism);
    }

    pub fn calculate_fitness<D>(&mut self, dataset: &D)
    where
        D: FitnessTrait,
    {
        for organism in &self.organisms {
            organism.set_fitness(dataset.calculate(organism));
        }

        self.cut_costs();
    }

    fn cut_costs(&mut self) {
        self.organisms.sort();
        self.organisms = self.organisms[0..self.size].to_vec();
    }

    pub fn epoch<D>(&mut self, dataset: &D) -> Option<&T>
    where
        D: FitnessTrait,
    {
        let size = self.organisms.len();

        for i in 0..size {
            let min_j = (size + get_random_range(0, size - 1)) % size;

            if let Ok(organism) = self.organisms[i].mutate(self.organisms.get(min_j), &self.config)
            {
                organism.set_fitness(dataset.calculate(&organism));
                self.add_organism(organism)
            }
        }

        self.cut_costs();

        match self.organisms.first() {
            Some(best) => {
                best.inc_stagnation();
                Some(best)
            }
            _ => None,
        }
    }
}
