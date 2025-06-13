use rand::distr::Alphanumeric;
use rand::{random, Rng};

/**
Generate unique id

```
use vivalaakam_neuro_utils::make_id;

let id = make_id(6);
assert_eq!(id.len(), 6)
```
 */
pub fn make_id(len: usize) -> String {
    rand::rng()
        .sample_iter(&Alphanumeric)
        .take(len)
        .map(char::from)
        .collect()
}

pub fn make_u8_id(len: usize) -> Vec<u8> {
    (0..len).map(|_| random::<u8>()).collect()
}
