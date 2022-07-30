use log::LevelFilter;

use vivalaakam_neat_rs::{Activation, Config, Genome, Organism};

fn get_fitness(organism: &mut Organism) {
    let mut distance: f64 = 0f64;
    let output = organism.activate(vec![0f64, 0f64]);
    distance += (0f64 - output[0]).powi(2);
    let output = organism.activate(vec![0f64, 1f64]);
    distance += (1f64 - output[0]).powi(2);
    let output = organism.activate(vec![1f64, 0f64]);
    distance += (1f64 - output[0]).powi(2);
    let output = organism.activate(vec![1f64, 1f64]);
    distance += (0f64 - output[0]).powi(2);

    organism.set_fitness(16f64 / (1f64 + distance));
}

fn main() {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Info)
        .is_test(true)
        .try_init();

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
        node_bias_prob: 0.15,
        node_bias_delta: 0.1,
        node_bias: 1.0,
        node_activation_prob: 0.15,
    };

    let genome = Genome::generate_genome(2, 1, vec![], Some(Activation::Sigmoid), &config);

    while population.len() < population_size {
        match genome.mutate_connection_weight(&config) {
            Some(genome) => {
                let mut organism = Organism::new(genome);
                get_fitness(&mut organism);
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
            let child = if i > 0 {
                population.get((i - 1) / 2)
            } else {
                None
            };

            match population[i].mutate(child, &config) {
                None => {}
                Some(organism) => new_population.push(organism),
            }
        }

        for organism in new_population.iter_mut() {
            get_fitness(organism);

            if organism.get_fitness() > 15.5 {
                best = Some(organism.clone());
            }
        }

        population = [population, new_population].concat();
        population.sort();
        population = population[0..population_size].to_vec();

        if let Some(best) = population.get_mut(0) {
            best.inc_stagnation();
            println!(
                "{epoch}: {:.8} {}",
                best.get_fitness(),
                best.get_stagnation()
            );
        }
        epoch += 1;
    }

    println!("{}", best.unwrap().genome.as_json());
}
