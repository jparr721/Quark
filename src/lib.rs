extern crate num_traits;

use std::ops::{Add, Mul};

pub struct Matrix<T> {
    data: Vec<Vec<T>>,
    nrows: usize,
    ncols: usize,
}

impl<T> Matrix<T> {
    fn new(data: Vec<Vec<T>>, nrows: usize, ncols: usize) -> Matrix<T> {
        Matrix {
            data: data,
            nrows: nrows,
            ncols: ncols,
        }
    }
}

pub fn transpose<T: Copy>(input: Vec<Vec<T>>, mut output: Vec<Vec<T>>) -> Matrix<T> {
    let nrows = input.len();
    let ncols = input[0].len();

    for i in 0..nrows {
        for j in 0..ncols {
            output[i][j] = input[j][i];
        }
    }

    Matrix {
        data: output,
        nrows: ncols,
        ncols: nrows,
    }
}

pub struct Vector<T> {
    data: Vec<T>,
    nrows: usize,
    ncols: usize,
}

impl<T> Vector<T> {
    fn new(data: Vec<T>, nrows: usize) -> Vector<T> {
        Vector {
            data: data,
            nrows: nrows,
            ncols: 1
        }
    }
}

fn dot<T>(first: Vec<T>, second: Vec<T>) -> T
where
    T: num_traits::Zero + Mul<T, Output=T> + Copy
{
    assert_eq!(first.len(), second.len());
    let mut sum = T::zero();

    for i in 0..first.len() {
        sum = sum + first[i] * second[i]
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn it_makes_matrix() {
        let data = vec![vec![0u8; 2]; 2];
        let mat = Matrix::new(data.clone(), 2, 2);

        assert_eq!(data, mat.data);
    }

    #[test]
    fn it_makes_vector() {
        let data = vec![0u8; 2];
        let vec = Vector::new(data.clone(), 2);

        assert_eq!(data, vec.data);
    }

    #[test]
    fn it_transposes_matrix() {
        let data = vec![vec![0, 0], vec![1, 1]];
        let comp = vec![vec![0, 1], vec![0, 1]];
        let output = vec![vec![0, 0], vec![0, 0]];
        let mat = transpose(data, output.clone());

        assert_eq!(comp, mat.data);
    }

    #[test]
    fn it_performs_a_dot_product() {
        let first = vec![-1, 0, 1];
        let second = vec![-1, 0, 1];

        assert_eq!(dot(first, second), 2);
    }
}
