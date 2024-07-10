use lazy_static::lazy_static;
use ndarray::Array2;
use tracing::{info, level_filters::LevelFilter};

use vivalaakam_neuro_neat::{Config, Genome, Organism};
use vivalaakam_neuro_shared::OrganismTrait;
use vivalaakam_neuro_utils::random::get_random_range;
use vivalaakam_neuro_utils::Activation;

lazy_static! {
    static ref INPUTS: Array2<f32> = Array2::from_shape_vec(
        (8, 3),
        vec![
            0f32, 0f32, 0f32, 0f32, 0f32, 1f32, 0f32, 1f32, 0f32, 0f32, 1f32, 1f32, 1f32, 0f32,
            0f32, 1f32, 0f32, 1f32, 1f32, 1f32, 0f32, 1f32, 1f32, 1f32,
        ]
    )
    .expect("");
    static ref OUTPUTS: Array2<f32> = Array2::from_shape_vec(
        (8, 1),
        vec![0f32, 1f32, 1f32, 0f32, 1f32, 0f32, 0f32, 1f32,]
    )
    .expect("");
}

fn get_fitness(organism: &mut Organism) {
    let output = organism.activate_matrix(&INPUTS);

    let distance = (OUTPUTS.clone() - output)
        .iter()
        .map(|row| (*row).powi(2))
        .sum::<f32>();
    organism.set_fitness(64f32 / (1f32 + distance));
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::INFO)
        .with_test_writer()
        .init();

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

    let genome = Genome::generate_genome(3, 1, vec![], Some(Activation::Sigmoid), &config)?;

    while population.len() < population_size {
        if let Ok(genome) = genome.mutate_connection_weight(&config) {
            let mut organism = Organism::new(genome);
            get_fitness(&mut organism);
            population.push(organism);
        }
    }

    population.sort();

    let mut best = None;
    let mut epoch = 0;
    while best.is_none() {
        let mut new_population = vec![];

        for i in 0..population.len() {
            let min_j =
                (population.len() + get_random_range(0, population.len() - 1)) % population.len();

            if let Ok(organism) = population[i].mutate(population.get(min_j), &config) {
                new_population.push(organism);
            }
        }

        for organism in new_population.iter_mut() {
            get_fitness(organism);

            if organism.get_fitness() > 64f32 * 0.995 {
                best = Some(organism.clone());
            }
        }

        population = [population, new_population].concat();
        population.sort();
        population = population[0..population_size].to_vec();

        if let Some(best) = population.get_mut(0) {
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
        info!("{}", best.genome.as_json());

        let data = best.genome.to_weights();

        let organism = Organism::new(Genome::from_weights(data));

        let fitness = organism.activate_matrix(&INPUTS);

        let shape = fitness.shape();

        for i in 0..shape[0] {
            let mut res = vec![];
            let mut equals = true;

            for j in 0..shape[1] {
                res.push(OUTPUTS[[i, j]].round());
                res.push(fitness[[i, j]].round());

                equals = equals && OUTPUTS[[i, j]].round() == fitness[[i, j]].round();
            }

            info!("{equals}, {res:?}");
        }
    }

    Ok(())
}
