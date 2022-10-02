use new_york_utils::Matrix;

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

    pub fn activate(&self, inputs: Vec<f64>) -> Vec<f64> {
        let mut state = vec![0f64; self.neurons.len()];
        for neuron in &self.neurons {
            match neuron.get_neuron_type() {
                NeuronType::Input => state[neuron.get_position()] = inputs[neuron.get_position()],
                _ => {
                    state[neuron.get_position()] = neuron.get_bias();

                    for connection in neuron.get_connections() {
                        state[neuron.get_position()] +=
                            state[connection.get_from()] * connection.get_weight()
                    }

                    state[neuron.get_position()] = neuron.activate(state[neuron.get_position()]);
                }
            }
        }

        state[state.len() - self.outputs..].to_vec()
    }

    pub fn activate_matrix(&self, matrix: &Matrix<f64>) -> Matrix<f64> {
        let rows_length = matrix.get_rows();
        let mut state = Matrix::new(self.neurons.len(), rows_length);

        for neuron in &self.neurons {
            match neuron.get_neuron_type() {
                NeuronType::Input => {
                    for i in 0..rows_length {
                        let _res = state.set(
                            neuron.get_position(),
                            i,
                            matrix.get(neuron.get_position(), i).unwrap_or_default(),
                        );
                    }
                }
                _ => {
                    for i in 0..rows_length {
                        let mut value = neuron.get_bias();

                        for connection in neuron.get_connections() {
                            value += state.get(connection.get_from(), i).unwrap_or_default()
                                * connection.get_weight();
                        }

                        let _ = state.set(neuron.get_position(), i, neuron.activate(value));
                    }
                }
            }
        }

        let mut response = Matrix::new(self.outputs, rows_length);
        let neurons_start = self.neurons.len() - self.outputs;
        for c in 0..self.outputs {
            for i in 0..rows_length {
                let _ = response.set(c, i, state.get(neurons_start + c, i).unwrap_or_default());
            }
        }

        response
    }
}
