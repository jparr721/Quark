use num_traits;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// The Matrix Struct is a size-aware 2d vector
#[derive(Debug, Clone)]
pub struct Matrix<T> {
    pub data: Vec<Vec<T>>,
    pub nrows: usize,
    pub ncols: usize,
}

impl<T> Matrix<T> {
    pub fn new(data: Vec<Vec<T>>) -> Matrix<T> {
        let rows = data.len();
        let cols = data[0].len();
        Matrix {
            data: data,
            nrows: rows,
            ncols: cols,
        }
    }

    /// Merges one matrix into another one vec by vec
    pub fn merge(&mut self, other: &mut Matrix<T>) -> Result<(), &'static str>
    where
        T: num_traits::Zero + Copy,
    {
        if other.clone().shape() != self.clone().shape() {
            return Err("Shape misaslignment");
        }

        for i in 0..self.data.len() {
            self.data[i].append(&mut other.data[i]);
        }

        Ok(())
    }

    pub fn shape(self) -> Vec<usize>
    where
        T: num_traits::Zero + Copy,
    {
        vec![self.nrows, self.ncols]
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

pub fn matmul<T>(first: Matrix<T>, second: Matrix<T>) -> Result<Matrix<T>, &'static str>
where
    T: num_traits::Zero + Mul<T, Output = T> + Add<T, Output = T> + Copy,
{
    if first.ncols != second.nrows {
        return Err("Row-Column mis-alignment");
    }

    let mut vec = vec![vec![T::zero(); first.nrows]; second.ncols];

    for i in 0..first.nrows {
        for j in 0..second.ncols {
            for k in 0..second.nrows {
                vec[i][j] = vec[i][j] + first.data[i][k] * second.data[k][j];
            }
        }
    }

    Ok(Matrix {
        data: vec,
        nrows: first.nrows,
        ncols: second.ncols,
    })
}

/// Performs a reduction operation on a given matrix, giving the reduced row echelon form
pub fn reduce<T: Eq>(mat: &mut Matrix<T>) -> Result<Matrix<T>, &'static str>
where
    T: num_traits::Zero
        + num_traits::One
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Neg<Output = T>
        + Div<T, Output = T>
        + Copy,
{
    let exchange = |matrix: &mut Matrix<T>, i: usize, j: usize| {
        matrix.data.swap(i, j);
    };

    let scale = |matrix: &mut Matrix<T>, row: usize, factor: T| {
        for i in 0..matrix.data[row].len() {
            matrix.data[row][i] = matrix.data[row][i] * factor;
        }
    };

    let row_replace = |matrix: &mut Matrix<T>, i: usize, j: usize, factor: T| {
        for k in 0..matrix.data[j].len() {
            matrix.data[j][k] = matrix.data[j][k] + (matrix.data[i][k] * factor);
        }
    };

    // Reduction steps
    let n = mat.data.len();

    for i in 0..n {
        // Find a pivot point
        for j in i..n {
            if mat.data[j][i] != T::zero() {
                if i != j {
                    exchange(mat, i, j);
                    break;
                }
            }

            if j == n - 1 {
                println!("col: {}", i);
                return Err("No pivot found");
            }
        }

        // Put zeros below diagonal
        for j in i + 1..n {
            row_replace(mat, i, j, -mat.data[j][i] / mat.data[i][i]);
        }
    }

    // Back substitution (bottom up)
    for i in (0..n - 1).rev() {
        for j in 0..i {
            row_replace(mat, i, j, -mat.data[j][i] / mat.data[i][i]);
        }
    }

    // Add 1's to the diagonal
    for i in 0..n {
        scale(mat, i, T::one() / mat.data[i][i]);
    }

    Ok(mat.clone())
}

#[test]
fn it_makes_matrix() {
    let data = vec![vec![0u8; 2]; 2];
    let mat = Matrix::new(data.clone());

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

    let res = matmul(data, data2).unwrap();

    assert_eq!(res.data, comp.data);
}

#[test]
fn it_row_reduces() {
    let mut mat = Matrix {
        data: vec![vec![2, 1, 4], vec![1, 2, 5]],
        nrows: 2,
        ncols: 3,
    };

    let comp = Matrix {
        data: vec![vec![1, 0, 1], vec![0, 1, 2]],
        nrows: 2,
        ncols: 3,
    };

    let res = reduce(&mut mat).unwrap();
    assert_eq!(res.data, comp.data);
}
