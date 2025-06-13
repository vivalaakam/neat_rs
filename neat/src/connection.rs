use serde::{Deserialize, Serialize};

/// Represents a connection (edge) between two nodes in the network.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    from: u32,
    to: u32,
    weight: f32,
    enabled: bool,
}

impl Connection {
    /// Creates a new connection between two nodes with a given weight.
    pub fn new(from: u32, to: u32, weight: f32) -> Self {
        Connection {
            from,
            to,
            weight,
            enabled: true,
        }
    }

    /// Toggles the enabled/disabled state of the connection.
    pub fn toggle_enabled(&mut self) {
        self.enabled = !self.enabled
    }

    /// Sets the enabled state of the connection.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether the connection is enabled.
    pub fn get_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns a unique string identifier for the connection.
    pub fn get_id(&self) -> String {
        format!("{}:{}", self.from, self.to)
    }

    /// Returns the source node ID.
    pub fn get_from(&self) -> u32 {
        self.from
    }

    /// Returns the target node ID.
    pub fn get_to(&self) -> u32 {
        self.to
    }

    /// Returns the connection's weight.
    pub fn get_weight(&self) -> f32 {
        self.weight
    }

    /// Sets the connection's weight.
    pub fn set_weight(&mut self, weight: f32) {
        self.weight = weight
    }

    /// Converts the connection to a vector of weights for serialization.
    pub fn to_weights(&self) -> Vec<f32> {
        let enabled = if self.enabled { 1u8 } else { 0u8 };

        let info = f32::from_le_bytes([enabled, 0, 0, 0]);

        vec![self.from as f32, self.to as f32, self.weight, info]
    }

    /// Creates a connection from a vector of weights.
    pub fn from_weights(weights: &mut dyn Iterator<Item = f32>) -> Self {
        let from = weights.next().expect("got not enough weights") as u32;
        let to = weights.next().expect("got not enough weights") as u32;
        let weight = weights.next().expect("got not enough weights");
        let info = weights.next().expect("got not enough weights");

        let enabled = info.to_le_bytes()[0] == 1u8;

        Connection {
            from,
            to,
            weight,
            enabled,
        }
    }
}
