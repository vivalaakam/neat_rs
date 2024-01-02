use ndarray::Array2;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LevenshteinError {
    #[error("Min value cannot be found")]
    MinNotFound,
}

pub fn levenshtein<T>(vec_1: Vec<T>, vec_2: Vec<T>) -> Result<i32, LevenshteinError>
where
    T: PartialEq,
{
    if vec_1.is_empty() {
        return Ok(vec_2.len() as i32);
    }
    if vec_2.is_empty() {
        return Ok(vec_1.len() as i32);
    }

    let vec_1_len = vec_1.len() + 1;
    let vec_2_len = vec_2.len() + 1;

    let mut matrix = Array2::<i32>::zeros((vec_2_len, vec_1_len));

    for i in 0..vec_2_len {
        matrix[[i, 0]] = i as i32;
    }

    for j in 0..vec_1_len {
        matrix[[0, j]] = j as i32;
    }

    for i in 1..vec_2_len {
        for j in 1..vec_1_len {
            let cost = if vec_2[i - 1] == vec_1[j - 1] { 0 } else { 1 };
            let val = *[
                matrix[[i - 1, j - 1]] + cost,
                matrix[[i, j - 1]] + 1,
                matrix[[i - 1, j]] + 1,
            ]
            .iter()
            .min()
            .ok_or(LevenshteinError::MinNotFound)?;

            matrix[[i, j]] = val;
        }
    }

    Ok(matrix[[vec_2.len(), vec_1.len()]])
}

#[cfg(test)]
mod tests {
    use crate::levenshtein::levenshtein;

    #[test]
    fn it_works() {
        let distance = levenshtein(
            vec!["k", "i", "t", "t", "e", "n"],
            vec!["s", "m", "i", "t", "t", "e", "n"],
        );

        assert_eq!(distance.unwrap(), 2);

        let distance = levenshtein(
            vec!["k", "i", "t", "t", "e", "n"],
            vec!["f", "i", "t", "t", "i", "n", "g"],
        );

        assert_eq!(distance.unwrap(), 3);
    }
}
