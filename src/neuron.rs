use crate::link::Link;
use crate::neuron_type::NeuronType;

#[derive(Default, Clone)]
pub struct Neuron {
    position: usize,
    bias: f64,
    neuron_type: NeuronType,
    connections: Vec<Link>,
}

impl Neuron {
    pub fn new(
        neuron_type: NeuronType,
        bias: f64,
        position: usize,
        connections: Vec<Link>,
    ) -> Self {
        Neuron {
            neuron_type,
            position,
            bias,
            connections,
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
}
