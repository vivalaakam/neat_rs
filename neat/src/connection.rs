use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    from: u32,
    to: u32,
    weight: f32,
    enabled: bool,
}

impl Connection {
    pub fn new(from: u32, to: u32, weight: f32) -> Self {
        Connection {
            from,
            to,
            weight,
            enabled: true,
        }
    }

    pub fn toggle_enabled(&mut self) {
        self.enabled = !self.enabled
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn get_enabled(&self) -> bool {
        self.enabled
    }

    pub fn get_id(&self) -> String {
        format!("{}:{}", self.from, self.to)
    }

    pub fn get_from(&self) -> u32 {
        self.from
    }

    pub fn get_to(&self) -> u32 {
        self.to
    }

    pub fn get_weight(&self) -> f32 {
        self.weight
    }

    pub fn set_weight(&mut self, weight: f32) {
        self.weight = weight
    }

    pub fn to_weights(&self) -> Vec<f32> {
        let enabled = if self.enabled { 1u8 } else { 0u8 };

        let info = f32::from_le_bytes([enabled, 0, 0, 0]);

        vec![self.from as f32, self.to as f32, self.weight, info]
    }

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
