use std::error::Error;

use ndarray::Array2;

pub struct OrganismTraitError {
    pub cause: Box<dyn Error>,
}

impl OrganismTraitError {
    pub fn new(cause: Box<dyn Error>) -> Self {
        OrganismTraitError { cause }
    }
}

pub trait OrganismTrait<C> {
    fn activate(&self, inputs: Vec<f32>) -> Vec<f32>;
    fn activate_matrix(&self, matrix: &Array2<f32>) -> Array2<f32>;
    fn set_fitness(&self, fitness: f32);
    fn get_fitness(&self) -> f32;
    fn inc_stagnation(&self);
    fn get_stagnation(&self) -> usize;
    fn mutate(&self, other: Option<&Self>, config: &C) -> Result<Self, OrganismTraitError>
    where
        Self: Sized;
}
