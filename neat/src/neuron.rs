use crate::link::Link;
use crate::neuron_type::NeuronType;
use crate::Node;
use vivalaakam_neuro_utils::Activation;

#[derive(Default, Clone)]
pub struct Neuron {
    position: u32,
    bias: f32,
    neuron_type: NeuronType,
    connections: Vec<Link>,
    activation: Activation,
}

impl From<Node> for Neuron {
    fn from(value: Node) -> Self {
        Neuron {
            position: value.get_position(),
            bias: value.get_bias(),
            neuron_type: value.get_type(),
            connections: vec![],
            activation: value.get_activation(),
        }
    }
}

impl Neuron {
    pub fn with_connections(self, connections: Vec<Link>) -> Self {
        Neuron {
            connections,
            ..self
        }
    }

    pub fn new(
        neuron_type: NeuronType,
        bias: f32,
        position: u32,
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

    pub fn get_bias(&self) -> f32 {
        self.bias
    }

    pub fn get_position(&self) -> u32 {
        self.position
    }

    pub fn get_neuron_type(&self) -> &NeuronType {
        &self.neuron_type
    }

    pub fn get_connections(&self) -> Vec<Link> {
        self.connections.to_vec()
    }

    pub fn activate(&self, value: f32) -> f32 {
        self.activation.activate(value)
    }
}
