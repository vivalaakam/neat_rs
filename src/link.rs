#[derive(Clone)]
pub struct Link {
    weight: f64,
    from_id: usize,
    to_id: usize,
}

impl Link {
    pub fn new(from_id: usize, to_id: usize, weight: f64) -> Self {
        Link {
            from_id,
            to_id,
            weight,
        }
    }

    pub fn get_weight(&self) -> f64 {
        self.weight
    }

    pub fn get_from(&self) -> usize {
        self.from_id
    }

    pub fn get_to(&self) -> usize {
        self.to_id
    }
}
