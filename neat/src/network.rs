use ndarray::Array2;

use crate::neuron::Neuron;
use crate::neuron_type::NeuronType;

#[derive(Default, Clone)]
pub struct Network {
    inputs: usize,
    outputs: usize,
    neurons: Vec<Neuron>,
}

impl Network {
    pub fn new(neurons: Vec<Neuron>) -> Self {
        let mut network = Network {
            neurons,
            ..Network::default()
        };

        for neuron in &network.neurons {
            match neuron.get_neuron_type() {
                NeuronType::Input => {
                    network.inputs += 1;
                }
                NeuronType::Output => {
                    network.outputs += 1;
                }
                _ => {}
            }
        }

        network
    }

    pub fn activate(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut state = vec![0f32; self.neurons.len()];
        for neuron in &self.neurons {
            match neuron.get_neuron_type() {
                NeuronType::Input => {
                    state[neuron.get_position() as usize] = inputs[neuron.get_position() as usize]
                }
                _ => {
                    let value = neuron.get_connections().iter().fold(
                        neuron.get_bias(),
                        |a, b| a + state[b.get_from() as usize] * b.get_weight(),
                    );

                    state[neuron.get_position() as usize] = neuron.activate(value);
                }
            }
        }

        state[state.len() - self.outputs..].to_vec()
    }

    pub fn activate_matrix(&self, matrix: &Array2<f32>) -> Array2<f32> {
        let rows_length = matrix.shape()[0];
        let mut state = Array2::from_elem((rows_length, self.neurons.len()), 0f32);

        for neuron in &self.neurons {
            match neuron.get_neuron_type() {
                NeuronType::Input => {
                    for i in 0..rows_length {
                        state[[i, neuron.get_position() as usize]] = matrix[[i, neuron.get_position() as usize]];
                    }
                }
                _ => {
                    for i in 0..rows_length {
                        let mut value = neuron.get_bias();

                        for connection in neuron.get_connections() {
                            value += state[[i, connection.get_from() as usize]]
                                * connection.get_weight();
                        }

                        state[[i, neuron.get_position() as usize]] = neuron.activate(value);
                    }
                }
            }
        }

        let mut response = Array2::from_elem((rows_length, self.outputs), 0f32);
        let neurons_start = self.neurons.len() - self.outputs;
        for c in 0..self.outputs {
            for i in 0..rows_length {
                response[[i, c]] = state[[i, neurons_start + c]];
            }
        }

        response
    }
}
