use new_york_utils::{levenshtein, Matrix};
use serde::Deserialize;
use tracing::{event, level_filters::LevelFilter, Level};

use vivalaakam_neat_rs::{Activation, Config, Genome, Organism};

fn get_fitness(organism: &mut Organism, inputs: &Matrix<f64>, outputs: &Matrix<f64>) {
    let results = organism.activate_matrix(inputs);
    let mut distance: f64 = 0f64;

    for i in 0..outputs.get_rows() {
        for j in 0..outputs.get_columns() {
            distance += (outputs.get(j, i).unwrap_or_default()
                - results.get(j, i).unwrap_or_default())
            .powi(2);
        }
    }

    event!(Level::DEBUG, "distance: {}", distance);

    organism.set_fitness((outputs.get_rows() * outputs.get_columns()) as f64 / (1f64 + distance));
}

#[derive(Debug, Deserialize)]
struct Record {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    variety_sentosa: f64,
    variety_versicolor: f64,
    variety_virginica: f64,
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::INFO)
        .with_test_writer()
        .init();

    let reader = csv::Reader::from_path("./examples/iris.csv");
    let mut inputs = Matrix::new(4, 0);
    let mut outputs = Matrix::new(3, 0);

    for record in reader.unwrap().deserialize::<Record>() {
        match record {
            Ok(rec) => {
                let _ = inputs.push_row(vec![
                    rec.sepal_length,
                    rec.sepal_width,
                    rec.petal_length,
                    rec.petal_width,
                ]);

                let _ = outputs.push_row(vec![
                    rec.variety_sentosa,
                    rec.variety_versicolor,
                    rec.variety_virginica,
                ]);

                event!(Level::INFO, "{rec:?}");
            }
            Err(_) => {}
        }
    }

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

    while population.len() < population_size * 4 {
        let genome = Genome::generate_genome(
            inputs.get_columns(),
            outputs.get_columns(),
            vec![],
            Some(Activation::Sigmoid),
            &config,
        );

        match genome.mutate_connection_weight(&config) {
            Some(genome) => {
                let mut organism = Organism::new(genome);
                get_fitness(&mut organism, &inputs, &outputs);
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
            get_fitness(organism, &inputs, &outputs);

            if organism.get_fitness() > 425f64 {
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

    if let Some(best) = best {
        let results = best.activate_matrix(&inputs);
        let mut success = 0;
        for i in 0..outputs.get_rows() {
            let mut res = vec![];
            let mut equals = true;

            for j in 0..outputs.get_columns() {
                res.push(outputs.get(j, i).unwrap_or_default().round());
                res.push(results.get(j, i).unwrap_or_default().round());

                equals = equals
                    && outputs.get(j, i).unwrap_or_default().round()
                        == results.get(j, i).unwrap_or_default().round();
            }

            if equals {
                success += 1;
            }

            event!(Level::INFO, "{equals}, {res:?}");
        }

        event!(Level::INFO, "{success}/{}", outputs.get_rows());
    }
}
