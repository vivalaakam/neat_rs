use rand::{random, Rng};
use rand::distributions::uniform::SampleUniform;

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
    rand::thread_rng().gen_range(from..to)
}
