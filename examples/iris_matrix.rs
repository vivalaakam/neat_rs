use new_york_utils::{levenshtein, Matrix};
use tracing::{event, Level, level_filters::LevelFilter};

use vivalaakam_neat_rs::{Activation, Config, Genome, Organism};

fn get_fitness(organism: &mut Organism, inputs: &Matrix<f64>) {
    let output = organism.activate_matrix(inputs);
    let mut distance: f64 = 0f64;
    let results = vec![0f64, 1f64, 1f64, 0f64];
    for i in 0..inputs.get_rows() {
        distance += (results[i] - output.get(0, i).unwrap_or_default()).powi(2);
    }

    organism.set_fitness(16f64 / (1f64 + distance));
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::INFO)
        .with_test_writer()
        .init();


    let population_size = 50;
    let mut population = vec![];

    let config = Config {
        add_node: 0.15,
        add_connection: 0.15,
        connection_enabled: 0.1,
        crossover: 0.3,
        connection_weight: 1.0,
        connection_weight_prob: 0.8,
        connection_weight_delta: 0.1,
        connection_weight_iter: 5,
        node_bias_prob: 0.15,
        node_bias_delta: 0.1,
        node_bias: 1.0,
        node_activation_prob: 0.15,
        connection_max: 10000,
        node_max: 1000,
        node_enabled: 0.5,
    };

    let genome = Genome::generate_genome(2, 1, vec![], Some(Activation::Sigmoid), &config);

    let mut inputs = Matrix::new(2, 4);
    inputs
        .set_data(vec![0f64, 0f64, 0f64, 1f64, 1f64, 0f64, 1f64, 1f64])
        .expect("TODO: panic message");

    while population.len() < population_size {
        match genome.mutate_connection_weight(&config) {
            Some(genome) => {
                let mut organism = Organism::new(genome);
                get_fitness(&mut organism, &inputs);
                population.push(organism);
            }
            _ => {}
        }
    }

    population.sort();

    let mut best = None;
    let mut epoch = 0;
    while best.is_none() {
        let mut new_population = vec![];

        for i in 0..population.len() {
            let mut child = None;

            let mut min_score = i32::MAX;
            let mut min_j = i;

            for j in i + 1..population.len() {
                let score = levenshtein(population[i].get_genotype(), population[j].get_genotype())
                    .unwrap_or(i32::MAX);

                if score > 0 && score < min_score {
                    min_score = score;
                    min_j = j;
                }
            }

            if min_j != i {
                child = population.get(min_j);
            }

            match population[i].mutate(child, &config) {
                None => {}
                Some(organism) => new_population.push(organism),
            }
        }

        for organism in new_population.iter_mut() {
            get_fitness(organism, &inputs);

            if organism.get_fitness() > 15.5 {
                best = Some(organism.clone());
            }
        }

        population = [population, new_population].concat();
        population.sort();
        population = population[0..population_size].to_vec();

        if let Some(best) = population.get_mut(0) {
            best.inc_stagnation();
            event!(
                Level::INFO,
                "{epoch}: {:.8} {}",
                best.get_fitness(),
                best.get_stagnation()
            );
        }
        epoch += 1;
    }

    event!(Level::INFO, "{}", best.unwrap().genome.as_json());
}
