use rand::Rng;

use vivalaakam_neuro_utils::Activation;

use crate::neuron::Neuron;

#[derive(Clone)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub activation: Activation,
}

impl Layer {
    pub fn new(neurons: Vec<Neuron>, activation: Activation) -> Self {
        assert!(!neurons.is_empty());

        assert!(neurons
            .iter()
            .all(|neuron| neuron.get_weights_size() == neurons[0].get_weights_size()));

        Self {
            neurons,
            activation,
        }
    }

    pub fn activate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.activate(&inputs))
            .collect()
    }

    pub fn from_weights(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        weights: &mut dyn Iterator<Item = f32>,
    ) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::from_weights(input_size, activation, weights))
            .collect();

        Self::new(neurons, activation)
    }

    pub fn get_weights(&self) -> Vec<f32> {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.get_weights())
            .collect()
    }

    pub fn get_topology(&self) -> Vec<usize> {
        vec![
            self.neurons.len(),
            u32::from_le_bytes([self.activation.to_bytes(), 0, 0, 0]) as usize,
        ]
    }

    pub fn random<T>(
        rng: &mut T,
        input_size: usize,
        output_size: usize,
        activation: Activation,
    ) -> Self
    where
        T: Rng,
    {
        let neurons = (0..output_size)
            .map(|_| Neuron::random(rng, activation, input_size))
            .collect();

        Self::new(neurons, activation)
    }
}
