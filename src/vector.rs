use std::ops::Mul;

/// The Vector Struct is a size-aware vector
pub struct Vector<T> {
    data: Vec<T>,
    nrows: usize,
    ncols: usize,
}

impl<T> Vector<T> {
    pub fn new(data: Vec<T>, nrows: usize) -> Vector<T> {
        Vector {
            data: data,
            nrows: nrows,
            ncols: 1,
        }
    }
}

pub fn dot<T>(first: Vector<T>, second: Vector<T>) -> T
where
    T: num_traits::Zero + Mul<T, Output = T> + Copy,
{
    assert_eq!(first.data.len(), second.data.len());
    let mut sum = T::zero();

    for i in 0..first.data.len() {
        sum = sum + first.data[i] * second.data[i]
    }
    sum
}

#[test]
fn it_makes_vector() {
    let data = vec![0u8; 2];
    let vec = Vector::new(data.clone(), 2);

    assert_eq!(data, vec.data);
}

#[test]
fn it_performs_a_dot_product() {
    let first = Vector {
        data: vec![-1, 0, 1],
        nrows: 3,
        ncols: 1,
    };
    let second = Vector {
        data: vec![-1, 0, 1],
        nrows: 3,
        ncols: 1,
    };

    assert_eq!(dot(first, second), 2);
}
