use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum NeuronType {
    Input,
    Hidden,
    Output,
    #[default]
    Unknown,
}

impl NeuronType {
    pub fn to_bytes(&self) -> u8 {
        match self {
            NeuronType::Input => 1,
            NeuronType::Hidden => 2,
            NeuronType::Output => 3,
            NeuronType::Unknown => 0,
        }
    }

    pub fn from_bytes(byte: u8) -> Self {
        match byte {
            1 => NeuronType::Input,
            2 => NeuronType::Hidden,
            3 => NeuronType::Output,
            _ => NeuronType::Unknown,
        }
    }
}
