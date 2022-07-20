use log::LevelFilter;

use vivalaakam_neat_rs::{Config, Genome, Organism};

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
        // Include all events in tests
        .filter_level(LevelFilter::Info)
        // Ensure events are captured by `cargo test`
        .is_test(true)
        // Ignore errors initializing the logger if tests race to configure it
        .try_init();

    let population_size = 50;
    let mut population = vec![];

    let config = Config {
        add_node: 0.15,
        add_connection: 0.15,
        connection_enabled: 0.1,
        crossover: 0.3,
        connection_weight: 0.8,
    };

    let genome = Genome::generate_genome(2, 1, vec![]);

    while population.len() < population_size {
        match genome.mutate_connection_weight() {
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
            let j = if i > 0 { (i - 1) / 2 } else {
                population.len() - 1
            };

            match population[i].mutate(&population[j], &config) {
                None => {}
                Some(organism) => { new_population.push(organism) }
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

        println!("{epoch}: {}", population[0].fitness);
        epoch += 1;
    }

    println!("{}", best.unwrap().genome.as_json());
}
