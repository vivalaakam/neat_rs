#[derive(Default)]
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
