use std::ops::{Mul, Add};
use num_traits;

/// The Vector Struct is a size-aware vector
#[derive(Debug, Clone)]
pub struct Vector<T> {
    pub data: Vec<T>,
    pub nrows: usize,
    pub ncols: usize,
}

impl<T> Vector<T> {
    pub fn new(data: Vec<T>) -> Vector<T> {
        let size = data.len();
        Vector {
            data: data,
            nrows: size,
            ncols: 1,
        }
    }

    pub fn append(&mut self, value: T) {
        let mut values_list = vec![value];
        self.data.append(&mut values_list);
        self.nrows += values_list.len();
    }

    pub fn shape(self) -> Vec<usize> {
        vec![self.nrows, self.ncols]
    }

    pub fn sum_vec(self, init: &T) -> Result<T, &'static str>
    where
        T: Copy + Add<T, Output = T>,
    {
        if self.data.len() <= 0 {
            return Err("Data must contain a value to be summed");
        }

        Ok(
            self.data.iter().fold(*init, |acc, &item| acc + item)
        )

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
    let vec = Vector::new(data.clone());

    assert_eq!(data, vec.data);
}

#[test]
fn it_gives_size() {
    let vec = Vector::new(vec![0, 1, 2, 3]);

    assert_eq!(vec![4, 1], vec.shape());
}

#[test]
fn it_sums_vector() {
    let vec = Vector::new(vec![0, 1, 2]);

    let sum = vec.sum_vec(&0).unwrap();

    assert_eq!(3, sum);
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
