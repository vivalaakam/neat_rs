use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    from: String,
    to: String,
    weight: f64,
    enabled: bool,
}

impl Connection {
    pub fn new<T>(from: T, to: T, weight: f64) -> Self
    where
        T: Into<String>,
    {
        Connection {
            from: from.into(),
            to: to.into(),
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

    pub fn get_from(&self) -> String {
        self.from.to_string()
    }

    pub fn get_to(&self) -> String {
        self.to.to_string()
    }

    pub fn get_weight(&self) -> f64 {
        self.weight
    }

    pub fn set_weight(&mut self, weight: f64) {
        self.weight = weight
    }
}
