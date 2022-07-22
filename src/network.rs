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
}
