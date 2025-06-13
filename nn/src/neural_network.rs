use rand::Rng;
use sha2::{Digest, Sha256};

use vivalaakam_neuro_utils::Activation;

use crate::layer::Layer;

#[derive(Clone)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn activate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.activate(inputs))
    }

    pub fn get_weights(&self) -> Vec<f32> {
        self.layers
            .iter()
            .flat_map(|layer| layer.get_weights())
            .collect()
    }

    pub fn get_hash(&self) -> String {
        let bytes = self
            .get_weights()
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect::<Vec<u8>>();
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let result = hasher.finalize();
        format!("{result:x}").to_lowercase()
    }

    pub fn get_topology_hash(&self) -> String {
        let bytes = self
            .get_topology()
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect::<Vec<u8>>();
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let result = hasher.finalize();
        format!("{result:x}").to_lowercase()
    }

    pub fn get_topology(&self) -> Vec<usize> {
        [
            vec![
                2,
                1,
                self.layers
                    .first()
                    .expect("got no layers")
                    .neurons
                    .first()
                    .expect("got no neurons")
                    .get_weights_size(),
                self.layers.len(),
            ],
            self.layers
                .iter()
                .flat_map(|layer| layer.get_topology())
                .collect(),
        ]
        .concat()
    }

    pub fn to_weights(&self) -> Vec<f32> {
        self.get_topology()
            .iter()
            .map(|&size| size as f32)
            .chain(self.get_weights())
            .collect()
    }

    pub fn from_weights(weights: impl IntoIterator<Item = f32>) -> Self {
        let mut weights = weights.into_iter();
        let mut inputs = weights.next().expect("got not enough weights") as usize;

        let layers = (0..weights.next().expect("got not enough layers") as usize)
            .map(|_| {
                let outputs = weights.next().expect("got not enough layers") as usize;
                let info = (weights.next().expect("got not enough layers") as u32).to_le_bytes();
                let activation = Activation::from_bytes(info[0]);
                let layer = Layer::from_weights(inputs, outputs, activation, &mut weights);
                inputs = outputs;
                layer
            })
            .collect::<Vec<_>>();

        assert!(!layers.is_empty());

        if weights.next().is_some() {
            panic!("got too many weights");
        }

        Self::new(layers)
    }

    pub fn random<T>(rng: &mut T, topology: &[usize]) -> Self
    where
        T: Rng,
    {
        let topology = &mut topology.iter().copied();
        let network_type = topology.next().expect("got not enough weights");
        assert_eq!(network_type, 2);

        let network_version = topology.next().expect("got not enough weights");
        assert_eq!(network_version, 1);

        let mut inputs = topology.next().expect("got not enough weights");

        let layers = (0..topology.next().expect("got not enough layers"))
            .map(|_| {
                let outputs = topology.next().expect("got not enough layers");
                let info = (topology.next().expect("got not enough layers") as u32).to_le_bytes();
                let activation = Activation::from_bytes(info[0]);
                let layer = Layer::random(rng, inputs, outputs, activation);
                inputs = outputs;
                layer
            })
            .collect();

        Self::new(layers)
    }
}

#[cfg(test)]
mod tests {
    use tracing::info;
    use tracing::level_filters::LevelFilter;

    use super::*;

    #[test]
    fn test_topology() {
        tracing_subscriber::fmt()
            .with_max_level(LevelFilter::INFO)
            .init();

        let mut rng = rand::rng();

        let topology = vec![2, 1, 2, 2, 2, 1, 1, 1];
        info!("{:?}", topology);
        let nn = NeuralNetwork::random(&mut rng, &topology);

        info!("{:?}", nn.to_weights());

        assert_eq!(nn.get_topology(), topology);
    }
}
