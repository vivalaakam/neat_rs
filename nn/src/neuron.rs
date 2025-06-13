use std::iter::once;

use rand::Rng;

use vivalaakam_neuro_utils::Activation;

#[derive(Clone)]
pub struct Neuron {
    bias: f32,
    weights: Vec<f32>,
    activation: Activation,
}

impl Neuron {
    pub fn new(bias: f32, activation: Activation, weights: Vec<f32>) -> Self {
        assert!(!weights.is_empty());

        Self {
            bias,
            weights,
            activation,
        }
    }

    pub fn get_weights_size(&self) -> usize {
        self.weights.len()
    }

    pub fn from_weights(
        input_size: usize,
        activation: Activation,
        weights: &mut dyn Iterator<Item = f32>,
    ) -> Self {
        let bias = weights.next().expect("got not enough weights");

        let weights = (0..input_size)
            .map(|_| weights.next().expect("got not enough weights"))
            .collect();

        Self::new(bias, activation, weights)
    }

    pub fn get_weights(&self) -> Vec<f32> {
        once(self.bias).chain(self.weights.clone()).collect()
    }

    pub fn random<T>(rng: &mut T, activation: Activation, input_size: usize) -> Self
    where
        T: Rng,
    {
        let bias = rng.random_range(-1.0..=1.0);
        let weights = (0..input_size).map(|_| rng.random_range(-1.0..=1.0)).collect();

        Self::new(bias, activation, weights)
    }

    pub fn activate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        self.activation.activate(self.bias + output)
    }
}
