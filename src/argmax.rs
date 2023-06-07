pub fn argmax(arr: Vec<f64>) -> usize {
    let mut maxi = 0;
    let mut max = &arr[0];

    for (i, item) in arr.iter().enumerate().skip(1) {
        if item > max {
            maxi = i;
            max = item;
        }
    }

    maxi
}
