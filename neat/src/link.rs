#[derive(Clone)]
pub struct Link {
    weight: f32,
    from_id: u32,
    to_id: u32,
}

impl Link {
    pub fn new(from_id: u32, to_id: u32, weight: f32) -> Self {
        Link {
            from_id,
            to_id,
            weight,
        }
    }

    pub fn get_weight(&self) -> f32 {
        self.weight
    }

    pub fn get_from(&self) -> u32 {
        self.from_id
    }

    pub fn get_to(&self) -> u32 {
        self.to_id
    }
}
