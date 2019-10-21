use num_traits;
use std::ops::{Add, Mul};

/// The Matrix Struct is a size-aware 2d vector
pub struct Matrix<T> {
    data: Vec<Vec<T>>,
    nrows: usize,
    ncols: usize,
}

impl<T> Matrix<T> {
    pub fn new(data: Vec<Vec<T>>, nrows: usize, ncols: usize) -> Matrix<T> {
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

pub fn matmul<T>(first: Matrix<T>, second: Matrix<T>) -> Matrix<T>
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

#[test]
fn it_makes_matrix() {
    let data = vec![vec![0u8; 2]; 2];
    let mat = Matrix::new(data.clone(), 2, 2);

    assert_eq!(data, mat.data);
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
