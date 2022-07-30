pub fn softmax(arr: Vec<f64>) -> Vec<f64> {
    let mut c = 0f64;
    for i in &arr {
        c = c.max(*i);
    }

    let d = arr
        .iter()
        .map(|y| (y - c).exp())
        .reduce(|a, b| a + b)
        .unwrap();
    arr.iter().map(|value| (value - c).exp() / d).collect()
}
