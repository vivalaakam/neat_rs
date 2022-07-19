use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub enum NeuronType {
    Input,
    Hidden,
    Output,
    Unknown,
}

impl Default for NeuronType {
    fn default() -> Self {
        NeuronType::Unknown
    }
}
