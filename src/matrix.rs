use num;
use num_traits;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// The Matrix Struct is a size-aware 2d vector
#[derive(Debug, Clone)]
pub struct Matrix<T: fmt::Display> {
    pub data: Vec<Vec<T>>,
    pub nrows: usize,
    pub ncols: usize,
}

impl<T: fmt::Display> Matrix<T> {
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

#[macro_export]
macro_rules! matrix {
    ($vec1:expr, $vec2:expr) => {{
        let mut temp_vec = Vec::new();
        temp_vec.push($vec1);
        temp_vec.push($vec2);
        let mat = Matrix::new(temp_vec);
        mat
    }};
}

impl<T: fmt::Display> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut comma_separated = String::new();
        comma_separated.push_str("[\n");

        for rowidx in 0..self.data.len() {
            comma_separated.push_str("\t");
            for colidx in 0..self.data[rowidx].len() {
                comma_separated.push_str(&self.data[rowidx][colidx].to_string());
                comma_separated.push_str(", ");
            }
            comma_separated.push_str("\n");
        }

        comma_separated.push_str("]");
        write!(f, "{}", comma_separated)
    }
}

pub fn t<T: fmt::Display>(input: Matrix<T>) -> Matrix<T>
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

pub fn matmul<T: fmt::Display>(
    first: Matrix<T>,
    second: Matrix<T>,
) -> Result<Matrix<T>, &'static str>
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

/// Performs a reduction operation on a given matrix via gauss-jordan elimination
pub fn gaussj<T: fmt::Display>(
    mat: &mut Matrix<T>,
    rhs: &mut Matrix<T>,
) -> Result<(Matrix<T>, Matrix<T>), &'static str>
where
    T: num_traits::Zero
        + num::Float
        + num_traits::One
        + num_traits::Signed
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Neg<Output = T>
        + Div<T, Output = T>
        + Copy,
{
    let n = mat.data.len();
    let mut irow = 0;
    let mut icol = 0;
    let mut pivinv = T::zero();

    let indxc = &mut vec![1; n];
    let indxr = &mut vec![1; n];
    let ipiv = &mut vec![T::zero(); n];

    for i in 0..n {
        let mut big = T::zero();
        // Find a pivot point
        for j in 0..n {
            if ipiv[j] != T::one() {
                for k in 0..n {
                    if mat.data[j][k].abs() >= big {
                        big = mat.data[j][k].abs();
                        irow = j;
                        icol = k;
                    }
                }
            } else if ipiv[n - 1] > T::one() {
                return Err("Singular matrix");
            }
        }
    }
    ipiv[icol] = ipiv[icol] + T::one();

    if irow != icol {
        for l in 0..n {
            let temp = mat.data[irow][l];
            mat.data[irow][l] = mat.data[icol][l];
            mat.data[icol][l] = temp;
        }

        for l in 0..n {
            let temp = rhs.data[irow][l];
            rhs.data[irow][l] = rhs.data[icol][l];
            rhs.data[icol][l] = temp;
        }
    }

    indxr[n - 1] = irow;
    indxc[n - 1] = icol;

    // Ensure that we aren't dealing with a singular matrix
    if mat.data[icol][icol] == T::zero() {
        return Err("Singular matrix");
    }

    // Take our pivot position
    pivinv = T::one() / mat.data[icol][icol];

    // Now, set the pivot spot to one
    mat.data[icol][icol] = T::one();

    // Now perform our scaling on the non-pivot data
    for l in 1..n {
        mat.data[icol][l] = mat.data[icol][l] * pivinv;
    }

    for l in 1..n {
        rhs.data[icol][l] = rhs.data[icol][l] * pivinv;
        println!("{}", rhs.to_string());
    }

    // Reduce the rows ignoring the pivot
    for ll in 1..n {
        if ll != icol {
            let dum = mat.data[ll][icol];
            for l in 1..n {
                mat.data[ll][l] = mat.data[ll][l] - mat.data[icol][l] * dum;
            }

            for l in 1..n {
                rhs.data[ll][l] = rhs.data[ll][l] - rhs.data[icol][l] * dum;
            }
        }
    }

    // Back substitution (bottom up approach)
    for l in (1..n).rev() {
        if indxr[l] != indxc[l] {
            for k in 1..n {
                let temp = mat.data[k][indxr[l]];
                mat.data[k][indxr[l]] = mat.data[k][indxc[l]];
                mat.data[k][indxc[l]] = temp;
            }
        }
    }

    Ok((mat.clone(), rhs.clone()))
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
        data: vec![vec![2.0, 1.0, 4.0], vec![1.0, 2.0, 5.0]],
        nrows: 2,
        ncols: 3,
    };

    let mut rhs = Matrix {
        data: vec![vec![0.0; 3], vec![0.0; 3]],
        nrows: 2,
        ncols: 3,
    };

    let comp = Matrix {
        data: vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 2.0]],
        nrows: 2,
        ncols: 3,
    };

    let res = gaussj(&mut mat, &mut rhs).unwrap();
    assert_eq!(res.0.data, comp.data);
}

#[test]
fn it_creates_from_macro() {
    let mat = matrix!(vec![0; 3], vec![0; 3]);
    let comp = Matrix::new(vec![vec![0; 3], vec![0; 3]]);

    assert_eq!(mat.data, comp.data);
}
