use serde::{Deserialize, Serialize};

use crate::neuron_type::NeuronType;

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Node {
    id: String,
    bias: f64,
    neuron_type: NeuronType,
    position: Option<usize>,
}

impl Node {
    pub fn new<T>(neuron_type: NeuronType, id: T, bias: f64, position: Option<usize>) -> Self
        where
            T: Into<String>,
    {
        Node {
            id: id.into(),
            neuron_type,
            position,
            bias,
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
}
