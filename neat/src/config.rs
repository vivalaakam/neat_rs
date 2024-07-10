#[derive(Clone)]
pub struct Config {
    pub add_node: f32,
    pub add_connection: f32,
    pub crossover: f32,
    pub connection_max: usize,
    pub connection_enabled: f32,
    pub connection_weight: f32,
    pub connection_weight_prob: f32,
    pub connection_weight_delta: f32,
    pub connection_weight_iter: usize,
    pub node_max: usize,
    pub node_bias: f32,
    pub node_enabled: f32,
    pub node_bias_prob: f32,
    pub node_bias_delta: f32,
    pub node_activation_prob: f32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            add_node: 0.15,
            add_connection: 0.15,
            connection_enabled: 0.1,
            crossover: 0.3,
            connection_weight: 1.0,
            connection_weight_prob: 0.8,
            connection_weight_delta: 0.1,
            connection_weight_iter: 5,
            node_bias_prob: 0.15,
            node_bias_delta: 0.1,
            node_bias: 1.0,
            node_activation_prob: 0.15,
            connection_max: 10000,
            node_max: 1000,
            node_enabled: 0.5,
        }
    }
}
