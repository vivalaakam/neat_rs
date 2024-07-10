use crate::organism::OrganismTrait;

pub trait FitnessTrait {
    fn calculate<T, C>(&self, organism: &T) -> f32
    where
        T: OrganismTrait<C>;
}
