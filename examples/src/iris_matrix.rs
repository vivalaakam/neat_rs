use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::Deserialize;
use tracing::{debug, info, level_filters::LevelFilter, warn};

use vivalaakam_neuro_neat::{Config, Genome, Organism};
use vivalaakam_neuro_shared::OrganismTrait;
use vivalaakam_neuro_utils::random::get_random_range;
use vivalaakam_neuro_utils::Activation;

fn get_fitness(organism: &mut Organism, inputs: &Array2<f32>, outputs: &Array2<f32>) {
    let results = organism.activate_matrix(inputs);
    let distance = (outputs - results)
        .iter()
        .map(|row| (*row).powi(2))
        .sum::<f32>();

    debug!("distance: {distance}");

    let shape = outputs.shape();
    organism.set_fitness((shape[0] * shape[1]) as f32 / (1f32 + distance));
}

#[derive(Debug, Deserialize)]
struct Record {
    sepal_length: f32,
    sepal_width: f32,
    petal_length: f32,
    petal_width: f32,
    variety_sentosa: f32,
    variety_versicolor: f32,
    variety_virginica: f32,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::INFO)
        .with_test_writer()
        .init();

    let reader = csv::Reader::from_path("examples/iris.csv");

    let mut inputs = vec![];
    let mut outputs = vec![];
    let mut total = 0;

    for record in reader.unwrap().deserialize::<Record>().flatten() {
        inputs = [
            inputs,
            vec![
                record.sepal_length,
                record.sepal_width,
                record.petal_length,
                record.petal_width,
            ],
        ]
        .concat();

        outputs = [
            outputs,
            vec![
                record.variety_sentosa,
                record.variety_versicolor,
                record.variety_virginica,
            ],
        ]
        .concat();

        total += 1;
    }
    let inputs_n = 4;
    let outputs_n = 3;

    let inputs = Array2::from_shape_vec((total, inputs_n), inputs).expect("");
    let outputs = Array2::from_shape_vec((total, outputs_n), outputs).expect("");

    let population_size = 50;
    let mut population = vec![];

    let config = Config {
        add_node: 0.10,
        add_connection: 0.25,
        connection_enabled: 0.25,
        crossover: 0.15,
        connection_weight: 2.0,
        connection_weight_prob: 0.8,
        connection_weight_delta: 0.35,
        connection_weight_iter: 25,
        node_bias_prob: 0.35,
        node_bias_delta: 0.35,
        node_bias: 2.0,
        node_activation_prob: 0.15,
        connection_max: 10000,
        node_max: 1000,
        node_enabled: 0.15,
    };

    while population.len() < population_size * 4 {
        let genome = Genome::generate_genome(
            inputs_n,
            outputs_n,
            vec![],
            Some(Activation::Sigmoid),
            &config,
        )?;

        if let Ok(genome) = genome.mutate_connection_weight(&config) {
            let mut organism = Organism::new(genome);
            get_fitness(&mut organism, &inputs, &outputs);
            population.push(organism);
        }
    }

    population.sort();

    let mut best = None;
    let mut epoch = 0;
    let mut best_genomes: Vec<Genome> = vec![];
    let mut bests_epoch = 0f32;
    let best_score = 100f32;
    while best.is_none() {
        let mut new_population = vec![];

        for i in 0..population.len() {
            let min_j =
                (population.len() + get_random_range(0, population.len() - 1)) % population.len();

            if let Ok(organism) = population[i].mutate(population.get(min_j), &config) {
                new_population.push(organism);
            }
        }

        let mut vec: Vec<usize> = (0..best_genomes.len()).collect();
        vec.shuffle(&mut thread_rng());

        for i in vec.iter().take(best_genomes.len() / 3) {
            if let Some(genome) = best_genomes.get(*i) {
                if let Ok(genome) = genome.mutate_connection_weight(&config) {
                    new_population.push(Organism::new(genome));
                }
            }
        }

        for organism in new_population.iter_mut() {
            get_fitness(organism, &inputs, &outputs);

            if organism.get_fitness() > best_score {
                best = Some(organism.clone());
            }
        }

        population = [population, new_population].concat();
        population.sort();
        population = population[0..population_size].to_vec();

        if let Some(best) = population.get_mut(0) {
            if best.get_stagnation() == 0 && best.get_fitness().round() > bests_epoch {
                warn!("add best genome");
                bests_epoch = best.get_fitness().round();
                best_genomes.push(best.genome.clone());
            }

            best.inc_stagnation();
            info!(
                "{epoch}: {:.8} {}",
                best.get_fitness(),
                best.get_stagnation()
            );
        }
        epoch += 1;
    }

    if let Some(best) = best {
        let results = best.activate_matrix(&inputs);
        let mut success = 0;
        let shape = results.shape();
        for i in 0..shape[0] {
            let mut res = vec![];
            let mut equals = true;

            for j in 0..shape[1] {
                res.push(outputs[[i, j]].round());
                res.push(results[[i, j]].round());

                equals = equals && outputs[[i, j]].round() == results[[i, j]].round();
            }

            if equals {
                success += 1;
            }

            info!("{equals}, {res:?}");
        }

        info!("{success}/{}", shape[0]);
    }

    Ok(())
}
