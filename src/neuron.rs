use crate::link::Link;
use crate::neuron_type::NeuronType;
use crate::Activation;

#[derive(Default, Clone)]
pub struct Neuron {
    position: usize,
    bias: f64,
    neuron_type: NeuronType,
    connections: Vec<Link>,
    activation: Activation,
}

impl Neuron {
    pub fn new(
        neuron_type: NeuronType,
        bias: f64,
        position: usize,
        activation: Activation,
        connections: Vec<Link>,
    ) -> Self {
        Neuron {
            neuron_type,
            position,
            bias,
            connections,
            activation,
        }
    }

    pub fn get_bias(&self) -> f64 {
        self.bias
    }

    pub fn get_position(&self) -> usize {
        self.position
    }

    pub fn get_neuron_type(&self) -> &NeuronType {
        &self.neuron_type
    }

    pub fn get_connections(&self) -> Vec<Link> {
        self.connections.to_vec()
    }

    pub fn activate(&self, value: f64) -> f64 {
        self.activation.activate(value)
    }
}
