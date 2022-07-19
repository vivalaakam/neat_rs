use rand::Rng;

pub fn get_random_position(len: usize) -> usize {
    rand::thread_rng().gen_range(0..len)
}

pub fn get_random_weight(weight: f64) -> f64 {
    rand::thread_rng().gen::<f64>() * (weight * 2.0) - weight
}

pub fn get_random() -> f32 {
    rand::thread_rng().gen()
}
