use rand::distr::uniform::SampleUniform;
use rand::{random, Rng};

pub fn get_random_position(len: usize) -> usize {
    get_random_range(0, len)
}

pub fn get_random_weight(weight: f32) -> f32 {
    get_random_range(-1f32 * weight, weight)
}

pub fn get_random() -> f32 {
    random()
}

pub fn get_random_range<T>(from: T, to: T) -> T
where
    T: SampleUniform + PartialOrd,
{
    rand::rng().random_range(from..to)
}
