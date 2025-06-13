use crate::neuron_type::NeuronType;
use serde::{Deserialize, Serialize};
use vivalaakam_neuro_utils::Activation;

/// Represents a neuron (node) in the network.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    id: u32,
    bias: f32,
    enabled: bool,
    activation: Activation,
    neuron_type: NeuronType,
    position: Option<u32>,
}

impl Node {
    /// Creates a new node with the given parameters.
    pub fn new(
        neuron_type: NeuronType,
        id: u32,
        bias: f32,
        activation: Option<Activation>,
        position: Option<u32>,
    ) -> Self {
        Node {
            id,
            neuron_type,
            enabled: true,
            position,
            bias,
            activation: activation.unwrap_or_default(),
        }
    }

    /// Returns the position of the node in the network.
    pub fn get_position(&self) -> u32 {
        self.position.unwrap_or_default()
    }

    /// Sets the position of the node in the network.
    pub fn set_position(&mut self, position: u32) {
        self.position = Some(position);
    }

    /// Returns the node's unique identifier.
    pub fn get_id(&self) -> u32 {
        self.id
    }

    /// Returns the type of the neuron (Input, Hidden, Output, etc).
    pub fn get_type(&self) -> NeuronType {
        self.neuron_type.clone()
    }

    /// Sets the node's bias value.
    pub fn set_bias(&mut self, bias: f32) {
        self.bias = bias;
    }

    /// Returns the node's bias value.
    pub fn get_bias(&self) -> f32 {
        self.bias
    }

    /// Sets the node's activation function.
    pub fn set_activation(&mut self, activation: Activation) {
        self.activation = activation;
    }

    /// Returns the node's activation function.
    pub fn get_activation(&self) -> Activation {
        self.activation
    }

    /// Toggles the enabled/disabled state of the node.
    pub fn toggle_enabled(&mut self) {
        self.enabled = !self.enabled
    }

    /// Returns whether the node is enabled.
    pub fn get_enabled(&self) -> bool {
        self.enabled
    }

    /// Converts the node to a vector of weights for serialization.
    pub fn to_weights(&self) -> Vec<f32> {
        let enabled = if self.enabled { 1u8 } else { 0u8 };

        let info = f32::from_le_bytes([
            enabled,
            self.activation.to_bytes(),
            self.neuron_type.to_bytes(),
            0,
        ]);

        vec![
            self.id as f32,
            self.bias,
            self.position.unwrap_or_default() as f32,
            info,
        ]
    }

    /// Creates a node from a vector of weights.
    pub fn from_weights(weights: &mut dyn Iterator<Item = f32>) -> Self {
        let id = weights.next().expect("got not enough weights") as u32;
        let bias = weights.next().expect("got not enough weights");
        let position = weights.next().expect("got not enough weights") as u32;

        let info = weights
            .next()
            .expect("got not enough weights")
            .to_le_bytes();

        let enabled = info[0] == 1;
        let activation = Activation::from_bytes(info[1]);
        let neuron_type = NeuronType::from_bytes(info[2]);

        Node {
            id,
            bias,
            enabled,
            activation,
            neuron_type,
            position: Some(position),
        }
    }
}
