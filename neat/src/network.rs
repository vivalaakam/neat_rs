use ndarray::Array2;

use vivalaakam_neuro_utils::Activation;

use crate::neuron::Neuron;
use crate::neuron_type::NeuronType;

#[derive(Clone)]
struct Step {
    position: usize,
    bias: f32,
    activation: Activation,
    start: usize,
    end: usize,
}

/// Feed-forward network compiled from a genome.
///
/// Neurons run in topological order. A batch is stored column-wise so each
/// connection is one contiguous multiply-add over every sample.
#[derive(Default, Clone)]
pub struct Network {
    inputs: usize,
    outputs: usize,
    neuron_count: usize,
    links_from: Vec<u32>,
    links_weight: Vec<f32>,
    steps: Vec<Step>,
}

impl Network {
    /// Creates a network from neurons already sorted in evaluation order.
    pub fn new(neurons: Vec<Neuron>) -> Self {
        let (inputs, outputs) =
            neurons
                .iter()
                .fold((0, 0), |(i, o), n| match n.get_neuron_type() {
                    NeuronType::Input => (i + 1, o),
                    NeuronType::Output => (i, o + 1),
                    _ => (i, o),
                });

        let mut links_from = Vec::new();
        let mut links_weight = Vec::new();
        let mut steps = Vec::new();

        for neuron in &neurons {
            if neuron.get_neuron_type() == &NeuronType::Input {
                continue;
            }

            let start = links_from.len();
            for link in neuron.connections() {
                links_from.push(link.get_from());
                links_weight.push(link.get_weight());
            }

            steps.push(Step {
                position: neuron.get_position() as usize,
                bias: neuron.get_bias(),
                activation: neuron.get_activation(),
                start,
                end: links_from.len(),
            });
        }

        Network {
            inputs,
            outputs,
            neuron_count: neurons.len(),
            links_from,
            links_weight,
            steps,
        }
    }

    fn eval_row(&self, state: &mut [f32]) {
        for step in &self.steps {
            let mut value = step.bias;
            for index in step.start..step.end {
                value += state[self.links_from[index] as usize] * self.links_weight[index];
            }
            state[step.position] = step.activation.activate(value);
        }
    }

    /// Activates the network with a single input vector.
    pub fn activate(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut state = vec![0f32; self.neuron_count];
        state[..self.inputs].copy_from_slice(&inputs[..self.inputs]);
        self.eval_row(&mut state);

        let output_at = self.neuron_count.saturating_sub(self.outputs);
        state[output_at..].to_vec()
    }

    /// Activates the network on a batch of inputs, one sample per row.
    pub fn activate_matrix(&self, matrix: &Array2<f32>) -> Array2<f32> {
        let rows = matrix.nrows();
        let width = self.neuron_count;
        let mut state = vec![0f32; rows * width];
        let mut acc = vec![0f32; rows];

        for column in 0..self.inputs {
            for row in 0..rows {
                state[column * rows + row] = matrix[[row, column]];
            }
        }

        for step in &self.steps {
            acc.fill(step.bias);
            for index in step.start..step.end {
                let from = self.links_from[index] as usize;
                let weight = self.links_weight[index];
                let column = &state[from * rows..(from + 1) * rows];
                for row in 0..rows {
                    acc[row] += column[row] * weight;
                }
            }

            let position = step.position;
            for row in 0..rows {
                state[position * rows + row] = step.activation.activate(acc[row]);
            }
        }

        let mut out = Vec::with_capacity(rows * self.outputs);
        let output_at = width.saturating_sub(self.outputs);
        for row in 0..rows {
            for column in 0..self.outputs {
                out.push(state[(output_at + column) * rows + row]);
            }
        }

        Array2::from_shape_vec((rows, self.outputs), out).expect("activate_matrix shape")
    }
}
