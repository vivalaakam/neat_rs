pub fn argmax(arr: Vec<f64>) -> usize {
    let mut maxi = 0;
    let mut max = arr[0];

    for i in 1..arr.len() {
        if arr[i] > max {
            maxi = i;
            max = arr[i];
        }
    }

    maxi
}
