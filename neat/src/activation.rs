use serde::{Deserialize, Serialize};
use strum::{EnumIter, IntoEnumIterator};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, EnumIter, Default)]
pub enum Activation {
    Sigmoid,
    Tanh,
    #[default]
    Identity,
    Step,
    Relu,
    SoftSign,
    Sinusoid,
    Gaussian,
    Selu,
}

impl Activation {
    pub fn activate(&self, x: f32) -> f32 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-4.924273 * x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Identity => x,
            Activation::Step => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Relu => x.max(0.0),
            Activation::SoftSign => x / (1.0 + x.abs()),
            Activation::Sinusoid => x.sin(),
            Activation::Gaussian => (-1.0 * x.powi(2)).exp(),
            Activation::Selu => {
                let alpha = 1.673_263_2;
                (if x > 0.0 { x } else { alpha * x.exp() - alpha }) * 1.050_700_9
            }
        }
    }

    pub fn to_vec() -> Vec<Activation> {
        Activation::iter().collect::<Vec<_>>()
    }
}
