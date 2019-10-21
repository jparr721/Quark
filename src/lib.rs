extern crate num_traits;

use std::ops::{Add, Mul};

/// The Matrix Struct is a size-aware 2d vector
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

pub fn t<T>(input: Matrix<T>) -> Matrix<T>
where
    T: num_traits::Zero + Copy,
{
    let mut vec = vec![vec![T::zero(); input.ncols]; input.nrows];

    for i in 0..input.nrows {
        for j in 0..input.ncols {
            vec[i][j] = input.data[j][i];
        }
    }

    Matrix {
        data: vec,
        nrows: input.ncols,
        ncols: input.nrows,
    }
}

fn matmul<T>(first: Matrix<T>, second: Matrix<T>) -> Matrix<T>
where
    T: num_traits::Zero + Mul<T, Output = T> + Add<T, Output = T> + Copy,
{
    assert_eq!(first.ncols, second.nrows);
    let mut vec = vec![vec![T::zero(); first.nrows]; second.ncols];

    for i in 0..first.nrows {
        for j in 0..second.ncols {
            for k in 0..second.nrows {
                vec[i][j] = vec[i][j] + first.data[i][k] * second.data[k][j];
            }
        }
    }

    Matrix {
        data: vec,
        nrows: first.nrows,
        ncols: second.ncols,
    }
}

/// The Vector Struct is a size-aware vector
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
            ncols: 1,
        }
    }
}

fn dot<T>(first: Vector<T>, second: Vector<T>) -> T
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
        let data = Matrix {
            data: vec![vec![0, 0], vec![1, 1]],
            nrows: 2,
            ncols: 2,
        };

        let comp = Matrix {
            data: vec![vec![0, 1], vec![0, 1]],
            nrows: 2,
            ncols: 2,
        };

        let mat = t(data);

        assert_eq!(comp.data, mat.data);
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

    #[test]
    fn it_matrix_multiplies() {
        let data = Matrix {
            data: vec![vec![1, 2], vec![2, 1]],
            nrows: 2,
            ncols: 2,
        };

        let data2 = Matrix {
            data: vec![vec![2, 1], vec![1, 2]],
            nrows: 2,
            ncols: 2,
        };

        let comp = Matrix {
            data: vec![vec![4, 5], vec![5, 4]],
            nrows: 2,
            ncols: 2,
        };

        assert_eq!(matmul(data, data2).data, comp.data);
    }
}
