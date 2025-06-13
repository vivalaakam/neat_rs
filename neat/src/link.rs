#[derive(Clone)]
/// Represents a weighted connection (link) between two nodes.
pub struct Link {
    weight: f32,
    from_id: u32,
    to_id: u32,
}

impl Link {
    /// Creates a new link between two nodes with a given weight.
    pub fn new(from_id: u32, to_id: u32, weight: f32) -> Self {
        Link {
            from_id,
            to_id,
            weight,
        }
    }

    /// Returns the weight of the link.
    pub fn get_weight(&self) -> f32 {
        self.weight
    }

    /// Returns the source node ID.
    pub fn get_from(&self) -> u32 {
        self.from_id
    }

    /// Returns the target node ID.
    pub fn get_to(&self) -> u32 {
        self.to_id
    }
}
