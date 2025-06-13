use std::ops::{Add, Mul};

use ndarray::{concatenate, s, Array1, Array2, Axis};
use tracing::debug;

use crate::neuron::Neuron;
use crate::neuron_type::NeuronType;

/// Represents a computational neural network built from a genome.
#[derive(Default, Clone)]
pub struct Network {
    inputs: usize,
    outputs: usize,
    neurons: Vec<Neuron>,
}

impl Network {
    /// Creates a new network from a list of neurons.
    pub fn new(neurons: Vec<Neuron>) -> Self {
        let (inputs, outputs) =
            neurons
                .iter()
                .fold((0, 0), |(i, o), n| match n.get_neuron_type() {
                    NeuronType::Input => (i + 1, o),
                    NeuronType::Output => (i, o + 1),
                    _ => (i, o),
                });

        Network {
            neurons,
            inputs,
            outputs,
        }
    }

    /// Activates the network with a single input vector.
    pub fn activate(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut state = vec![0f32; self.neurons.len()];
        for neuron in &self.neurons {
            match neuron.get_neuron_type() {
                NeuronType::Input => {
                    state[neuron.get_position() as usize] = inputs[neuron.get_position() as usize]
                }
                _ => {
                    let value = neuron
                        .get_connections()
                        .iter()
                        .fold(neuron.get_bias(), |a, b| {
                            a + state[b.get_from() as usize] * b.get_weight()
                        });

                    state[neuron.get_position() as usize] = neuron.activate(value);
                }
            }
        }

        state[state.len() - self.outputs..].to_vec()
    }

    /// Activates the network with a batch (matrix) of inputs.
    pub fn activate_matrix(&self, matrix: &Array2<f32>) -> Array2<f32> {
        let rows_length = matrix.shape()[0];
        let mut state = concatenate(
            Axis(1),
            &[
                matrix.view(),
                Array2::zeros((rows_length, self.neurons.len() - self.inputs)).view(),
            ],
        )
        .expect("");

        debug!("view: {:?}", state.view());

        for neuron in &self.neurons {
            match neuron.get_neuron_type() {
                NeuronType::Input => {}
                _ => {
                    let value = neuron.get_connections().iter().fold(
                        Array1::from_elem(rows_length, neuron.get_bias()),
                        |a, b| {
                            a.add(
                                &state
                                    .column(b.get_from() as usize)
                                    .mul(Array1::from_elem(rows_length, b.get_weight())),
                            )
                        },
                    );

                    state
                        .column_mut(neuron.get_position() as usize)
                        .assign(&value.map(|x| neuron.activate(*x)));
                }
            }
        }

        state.slice(s![.., -(self.outputs as i32)..]).to_owned()
    }
}
