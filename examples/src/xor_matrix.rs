use lazy_static::lazy_static;
use ndarray::Array2;
use tracing::{info, level_filters::LevelFilter};

use vivalaakam_neuro_neat::{Config, Genome, Organism};
use vivalaakam_neuro_shared::{FitnessTrait, OrganismTrait, Population};
use vivalaakam_neuro_utils::Activation;

lazy_static! {
    static ref INPUTS: Array2<f32> =
        Array2::from_shape_vec((4, 2), vec![0f32, 0f32, 0f32, 1f32, 1f32, 0f32, 1f32, 1f32])
            .expect("");
    static ref OUTPUTS: Array2<f32> =
        Array2::from_shape_vec((4, 1), vec![0f32, 1f32, 1f32, 0f32]).expect("");
}

struct Dataset {
    inputs: Array2<f32>,
    outputs: Array2<f32>,
}

impl FitnessTrait for Dataset {
    fn calculate<T, C>(&self, organism: &T) -> f32
    where
        T: OrganismTrait<C>,
    {
        let output = organism.activate_matrix(&self.inputs);

        let distance = (&self.outputs - output)
            .iter()
            .map(|row| (*row).powi(2))
            .sum::<f32>();

        16f32 / (1f32 + distance)
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::INFO)
        .with_test_writer()
        .init();

    let dataset = Dataset {
        inputs: INPUTS.clone(),
        outputs: OUTPUTS.clone(),
    };

    let population_size = 50;

    let config = Config::default();

    let mut population: Population<Config, Organism> =
        Population::new(population_size, config.clone());

    let genome = Genome::generate_genome(2, 1, vec![], Some(Activation::Sigmoid), &config)?;

    while population.len() < population_size {
        if let Ok(genome) = genome.mutate_connection_weight(&config) {
            population.add_organism(Organism::new(genome));
        }
    }

    population.calculate_fitness(&dataset);

    let mut best = None;
    let mut epoch = 0;
    while best.is_none() {
        if let Some(organism) = population.epoch(&dataset) {
            if organism.get_fitness() > 15.5 {
                best = Some(organism.clone());
            }

            info!(
                "{epoch}: {:.8} {}",
                organism.get_fitness(),
                organism.get_stagnation()
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
