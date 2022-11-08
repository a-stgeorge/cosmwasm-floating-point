mod float;

use float::Float;

fn main() {
    let pi: Float = Float::new(3141592653589793, 0).unwrap();
    println!("PI! {}", pi);
}


