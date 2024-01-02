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

    pub fn to_bytes(&self) -> u8 {
        match self {
            Activation::Sigmoid => 1,
            Activation::Tanh => 2,
            Activation::Identity => 3,
            Activation::Step => 4,
            Activation::Relu => 5,
            Activation::SoftSign => 6,
            Activation::Sinusoid => 7,
            Activation::Gaussian => 8,
            Activation::Selu => 9,
        }
    }

    pub fn from_bytes(byte: u8) -> Self {
        match byte {
            1 => Activation::Sigmoid,
            2 => Activation::Tanh,
            3 => Activation::Identity,
            4 => Activation::Step,
            5 => Activation::Relu,
            6 => Activation::SoftSign,
            7 => Activation::Sinusoid,
            8 => Activation::Gaussian,
            9 => Activation::Selu,
            _ => Activation::Identity,
        }
    }
}
