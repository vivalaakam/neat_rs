use thiserror::Error;

#[derive(Error, Debug)]
pub enum LevenshteinError {
    #[error("Min value cannot be found")]
    MinNotFound,
}

pub fn levenshtein<T>(left: &[T], right: &[T]) -> Result<i32, LevenshteinError>
where
    T: PartialEq,
{
    if left.is_empty() {
        return Ok(right.len() as i32);
    }
    if right.is_empty() {
        return Ok(left.len() as i32);
    }

    let mut previous: Vec<i32> = (0..=right.len() as i32).collect();
    let mut current = vec![0; right.len() + 1];

    for (row, item) in left.iter().enumerate() {
        current[0] = (row + 1) as i32;
        for (column, other) in right.iter().enumerate() {
            let cost = if item == other { 0 } else { 1 };
            let insert = current[column] + 1;
            let delete = previous[column + 1] + 1;
            let replace = previous[column] + cost;
            current[column + 1] = insert.min(delete).min(replace);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    Ok(previous[right.len()])
}

#[cfg(test)]
mod tests {
    use crate::levenshtein::levenshtein;

    #[test]
    fn it_works() {
        let distance = levenshtein(
            &["k", "i", "t", "t", "e", "n"],
            &["s", "m", "i", "t", "t", "e", "n"],
        );

        assert_eq!(distance.unwrap(), 2);

        let distance = levenshtein(
            &["k", "i", "t", "t", "e", "n"],
            &["f", "i", "t", "t", "i", "n", "g"],
        );

        assert_eq!(distance.unwrap(), 3);
    }
}
