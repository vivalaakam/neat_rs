/// Configuration parameters for NEAT evolution and mutation.
#[derive(Clone)]
pub struct Config {
    /// Probability of adding a node during mutation.
    pub add_node: f32,
    /// Probability of adding a connection during mutation.
    pub add_connection: f32,
    /// Probability of crossover during reproduction.
    pub crossover: f32,
    /// Maximum number of connections allowed.
    pub connection_max: usize,
    /// Probability of enabling/disabling a connection during mutation.
    pub connection_enabled: f32,
    /// Initial connection weight range.
    pub connection_weight: f32,
    /// Probability of mutating a connection's weight.
    pub connection_weight_prob: f32,
    /// Maximum delta for connection weight mutation.
    pub connection_weight_delta: f32,
    /// Number of iterations for connection weight mutation.
    pub connection_weight_iter: usize,
    /// Maximum number of nodes allowed.
    pub node_max: usize,
    /// Initial node bias range.
    pub node_bias: f32,
    /// Probability of enabling/disabling a node during mutation.
    pub node_enabled: f32,
    /// Probability of mutating a node's bias.
    pub node_bias_prob: f32,
    /// Maximum delta for node bias mutation.
    pub node_bias_delta: f32,
    /// Probability of mutating a node's activation function.
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
