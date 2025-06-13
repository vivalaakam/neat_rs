use serde::{Deserialize, Serialize};

/// Enum representing the type of a neuron in the network.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum NeuronType {
    Input,
    Hidden,
    Output,
    #[default]
    Unknown,
}

impl NeuronType {
    /// Converts the neuron type to a byte value.
    pub fn to_bytes(&self) -> u8 {
        match self {
            NeuronType::Input => 1,
            NeuronType::Hidden => 2,
            NeuronType::Output => 3,
            NeuronType::Unknown => 0,
        }
    }

    /// Creates a neuron type from a byte value.
    pub fn from_bytes(byte: u8) -> Self {
        match byte {
            1 => NeuronType::Input,
            2 => NeuronType::Hidden,
            3 => NeuronType::Output,
            _ => NeuronType::Unknown,
        }
    }
}
