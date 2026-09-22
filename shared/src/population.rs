use vivalaakam_neuro_utils::random::get_random_range;

use crate::fitness::FitnessTrait;
use crate::organism::OrganismTrait;

pub struct Population<C, T> {
    size: usize,
    organisms: Vec<T>,
    config: C,
}

fn worker_count(len: usize) -> usize {
    if len < 8 {
        return 1;
    }

    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, len)
}

fn run_chunks<R>(len: usize, work: impl Fn(usize, usize) -> R + Sync) -> Vec<R>
where
    R: Send,
{
    let workers = worker_count(len);
    if workers == 1 {
        return vec![work(0, len)];
    }

    let chunk = len.div_ceil(workers);
    std::thread::scope(|scope| {
        let work = &work;
        let mut handles = Vec::new();
        for worker in 0..workers {
            let start = worker * chunk;
            if start >= len {
                break;
            }
            let end = (start + chunk).min(len);
            handles.push(scope.spawn(move || work(start, end)));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("fitness worker panicked"))
            .collect()
    })
}

impl<C, T> Population<C, T>
where
    C: Clone + Default,
    T: OrganismTrait<C> + std::cmp::Ord + Clone,
{
    pub fn new(size: usize, config: C) -> Self {
        Population {
            size,
            config,
            organisms: vec![],
        }
    }

    pub fn len(&self) -> usize {
        self.organisms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.organisms.is_empty()
    }

    pub fn add_organism(&mut self, organism: T) {
        self.organisms.push(organism);
    }

    pub fn calculate_fitness<D>(&mut self, dataset: &D)
    where
        D: FitnessTrait + Sync,
        T: Sync,
    {
        let organisms = &self.organisms;
        run_chunks(organisms.len(), |start, end| {
            for organism in &organisms[start..end] {
                organism.set_fitness(dataset.calculate(organism));
            }
        });

        self.cut_costs();
    }

    fn cut_costs(&mut self) {
        self.organisms.sort();
        self.organisms.truncate(self.size);
    }

    pub fn epoch<D>(&mut self, dataset: &D) -> Option<&T>
    where
        D: FitnessTrait + Sync,
        T: Send + Sync,
        C: Sync,
    {
        let size = self.organisms.len();
        if size == 0 {
            return None;
        }

        let partners: Vec<usize> = (0..size)
            .map(|_| (size + get_random_range(0, size - 1)) % size)
            .collect();

        let children = run_chunks(size, |start, end| {
            let mut born = Vec::with_capacity(end - start);
            for (offset, partner) in partners[start..end].iter().enumerate() {
                let index = start + offset;
                if let Ok(organism) =
                    self.organisms[index].mutate(self.organisms.get(*partner), &self.config)
                {
                    organism.set_fitness(dataset.calculate(&organism));
                    born.push(organism);
                }
            }
            born
        });

        for born in children {
            self.organisms.extend(born);
        }

        self.cut_costs();

        match self.organisms.first() {
            Some(best) => {
                best.inc_stagnation();
                Some(best)
            }
            _ => None,
        }
    }
}
