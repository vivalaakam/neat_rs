use serde::{Deserialize, Serialize};
use strum::{EnumIter, IntoEnumIterator};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, EnumIter)]
pub enum Activation {
    Sigmoid,
    Tanh,
    Identity,
    Step,
    Relu,
    SoftSign,
    Sinusoid,
    Gaussian,
    Selu,
}

impl Default for Activation {
    fn default() -> Self {
        Activation::Identity
    }
}

impl Activation {
    pub fn activate(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => 1f64 / (1f64 + (-4.924273 * x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Identity => x,
            Activation::Step => {
                if x > 0f64 {
                    1f64
                } else {
                    0f64
                }
            }
            Activation::Relu => x.max(0f64),
            Activation::SoftSign => x / (1f64 + x.abs()),
            Activation::Sinusoid => x.sin(),
            Activation::Gaussian => (-1f64 * x.powi(2)).exp(),
            Activation::Selu => {
                let alpha = 1.6732632423543772848170429916717;
                (if x > 0f64 { x } else { alpha * x.exp() - alpha })
                    * 1.0507009873554804934193349852946
            }
        }
    }

    pub fn to_vec() -> Vec<Activation> {
        Activation::iter().collect::<Vec<_>>()
    }
}
