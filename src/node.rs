use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::neuron_type::NeuronType;

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    id: String,
    bias: f64,
    enabled: bool,
    activation: Activation,
    neuron_type: NeuronType,
    position: Option<usize>,
}

impl Node {
    pub fn new<T>(
        neuron_type: NeuronType,
        id: T,
        bias: f64,
        activation: Option<Activation>,
        position: Option<usize>,
    ) -> Self
    where
        T: Into<String>,
    {
        Node {
            id: id.into(),
            neuron_type,
            enabled: true,
            position,
            bias,
            activation: activation.unwrap_or_default(),
        }
    }

    pub fn get_position(&self) -> usize {
        self.position.unwrap_or_default()
    }

    pub fn set_position(&mut self, position: usize) {
        self.position = Some(position);
    }

    pub fn get_id(&self) -> String {
        self.id.to_string()
    }

    pub fn get_type(&self) -> NeuronType {
        self.neuron_type.clone()
    }

    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }

    pub fn get_bias(&self) -> f64 {
        self.bias
    }

    pub fn set_activation(&mut self, activation: Activation) {
        self.activation = activation;
    }

    pub fn get_activation(&self) -> Activation {
        self.activation
    }

    pub fn toggle_enabled(&mut self) {
        self.enabled = !self.enabled
    }

    pub fn get_enabled(&self) -> bool {
        self.enabled
    }
}
