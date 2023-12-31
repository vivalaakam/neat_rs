pub fn softmax(arr: Vec<f32>) -> Vec<f32>  {
    let c = &arr.iter().fold(0f32, |a, b| a.max(*b));

    let d = arr
        .iter()
        .map(|y| (y - c).exp())
        .reduce(|a, b| a + b)
        .unwrap_or(0f32);

    arr.iter().map(|value| (value - c).exp() / d).collect()
}
