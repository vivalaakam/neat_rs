pub use activation::Activation;
pub use argmax::argmax;
pub use levenshtein::levenshtein;
pub use make_id::{make_id, make_u8_id};
pub use softmax::softmax;

mod activation;
mod argmax;
mod levenshtein;
mod make_id;
pub mod random;
mod softmax;
