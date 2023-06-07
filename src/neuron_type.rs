use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum NeuronType {
    Input,
    Hidden,
    Output,
    #[default]
    Unknown,
}
