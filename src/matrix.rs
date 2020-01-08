use std::fmt;
use std::ops::Mul;

/// The Matrix Struct is a size-aware 2d vector
#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<Vec<f64>>) -> Matrix {
        Matrix {
            data: data,
            rows: rows,
            cols: cols,
        }
    }

    pub fn from_mat(rows: &usize, cols: &usize, data: &Vec<Vec<f64>>) -> Matrix {
        Matrix {
            data: data.clone(),
            rows: *rows,
            cols: *cols,
        }
    }

    pub fn data(&self) -> &Vec<Vec<f64>> {
        &self.data
    }

    pub fn mut_data(&mut self) -> &mut Vec<Vec<f64>> {
        &mut self.data
    }

    pub fn as_ptr(&self) -> *const Vec<f64> {
        self.data.as_ptr()
    }

    pub fn into_vec(self) -> Vec<Vec<f64>> {
        self.data
    }

    pub fn rows(self) -> usize {
        self.rows
    }

    pub fn cols(self) -> usize {
        self.cols
    }

    pub fn shape(self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows: rows,
            cols: cols,
            data: vec![vec![0.0; rows]; cols],
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows: rows,
            cols: cols,
            data: vec![vec![1.0; rows]; cols],
        }
    }

    pub fn len(self) -> usize {
        self.data.len()
    }

    pub fn mean_matrix(self) -> Result<Option<Matrix>, &'static str> {
        if self.rows < 1 {
            return Err("Matrix height must be greater than 0");
        }

        let mut out = Matrix::new(self.rows, self.cols, vec![Vec::with_capacity(1); self.cols]);

        for i in 0..self.cols {
            let mut col = vec![0.0; self.rows];
            for j in 0..self.rows {
                col.push(self.data[j][i]);
            }

            let mean = col.iter().fold(0.0, |a, &b| a + b) / col.len() as f64;
            out.data[0].push(mean);
        }

        Ok(Some(out))
    }

    pub fn t(&mut self) -> Result<Matrix, &'static str> {
        if self.rows < 1 || self.cols < 1 {
            return Err("Data must be present to transpose");
        }

        let mut t = vec![Vec::with_capacity(self.rows); self.cols];

        for r in &mut self.data {
            for i in 0..self.rows {
                t[i].push(r[i]);
            }
        }

        Ok(Matrix::new(self.rows, self.cols, t))
    }

    pub fn swap(&mut self, p: usize, q: usize) {
        assert!(
            self.rows >= 2,
            "There must be at least 2 rows in order to swap"
        );

        assert!(
            p < self.rows && q < self.rows,
            "You cannot specify a p less than the matrix height"
        );

        self.data.swap(p, q);
    }
}

impl Mul for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Matrix {
        assert_eq!(self.cols, rhs.rows);

        let mut vec = vec![vec![0.0; self.rows]; rhs.cols];

        for i in 0..self.rows {
            for j in 0..rhs.cols {
                for k in 0..rhs.rows {
                    vec[i][j] = vec[i][j] + self.data[i][k] * rhs.data[k][j];
                }
            }
        }

        Matrix {
            data: vec,
            rows: self.rows,
            cols: rhs.cols,
        }
    }
}

impl fmt::Display for Matrix {
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

pub fn row_replacement(matrix: &mut Matrix, i: usize, j: usize, factor: f64) {
    for k in 0..matrix.data[j].len() {
        matrix.data[j][k] += matrix.data[i][k] * factor;
    }
}

pub fn scale_row(matrix: &mut Matrix, row: usize, factor: f64) {
    let n = matrix.cols;

    for i in 0..n {
        matrix.data[row][i] *= factor;
    }
}

pub fn upper_triangular(matrix: &mut Matrix) -> Result<(), &'static str> {
    let n = matrix.clone().rows();

    for i in 0..n {
        // Find a pivot point
        for j in i..n {
            if matrix.data[j][i] != 0.0 {
                if i != j {
                    matrix.swap(i, j);
                }
                break;
            }

            if j == n - 1 {
                return Err("No pivots found in matrixrix!");
            }
        }

        for j in i + 1..n {
            row_replacement(matrix, i, j, -matrix.data[j][i] / matrix.data[i][i]);
        }
    }

    Ok(())
}

pub fn rowreduce(matrix: &mut Matrix) -> Result<(), &'static str> {
    let n = matrix.clone().rows();

    for i in 0..n {
        // Find a pivot point
        for j in i..n {
            if matrix.data[j][i] != 0.0 {
                if i != j {
                    matrix.swap(i, j);
                }
                break;
            }

            if j == n - 1 {
                return Err("No pivots found in matrixrix!");
            }
        }

        for j in i + 1..n {
            row_replacement(matrix, i, j, -matrix.data[j][i] / matrix.data[i][i]);
        }
    }

    // Back subsitution
    for i in (0..n).rev() {
        for j in 0..i {
            row_replacement(matrix, i, j, -matrix.data[j][i] / matrix.data[i][i]);
        }
    }

    // Ones along diagonal
    for i in 0..n {
        scale_row(matrix, i, 1.0 / matrix.data[i][i]);
    }

    Ok(())
}

#[test]
fn it_shows_shape() {
    let mat = Matrix::new(2, 3, vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]);

    assert_eq!(mat.shape(), (2, 3));
}

#[test]
fn it_makes_matrix() {
    let data = vec![vec![0.0; 2]; 2];
    let mat = Matrix::new(2, 2, data.clone());

    assert_eq!(data, mat.data);
}

#[test]
fn it_transposes_matrix() {
    let mut data = Matrix {
        data: vec![vec![0.0, 0.0], vec![1.0, 1.0]],
        rows: 2,
        cols: 2,
    };

    let comp = Matrix {
        data: vec![vec![0.0, 1.0], vec![0.0, 1.0]],
        rows: 2,
        cols: 2,
    };

    let res = data.t().unwrap();

    assert_eq!(comp.data, res.data);
}

#[test]
fn it_matrix_multiplies() {
    let data = Matrix {
        data: vec![vec![1.0, 2.0], vec![2.0, 1.0]],
        rows: 2,
        cols: 2,
    };

    let data2 = Matrix {
        data: vec![vec![2.0, 1.0], vec![1.0, 2.0]],
        rows: 2,
        cols: 2,
    };

    let comp = Matrix {
        data: vec![vec![4.0, 5.0], vec![5.0, 4.0]],
        rows: 2,
        cols: 2,
    };

    let res = matmul(data, data2).unwrap();

    assert_eq!(res.data, comp.data);
}

#[test]
fn it_overrides_matrix_multiply() {
    let data = Matrix {
        data: vec![vec![1.0, 2.0], vec![2.0, 1.0]],
        rows: 2,
        cols: 2,
    };

    let data2 = Matrix {
        data: vec![vec![2.0, 1.0], vec![1.0, 2.0]],
        rows: 2,
        cols: 2,
    };

    let comp = Matrix {
        data: vec![vec![4.0, 5.0], vec![5.0, 4.0]],
        rows: 2,
        cols: 2,
    };

    let res = data * data2;

    assert_eq!(res.data, comp.data);
}

#[test]
fn it_row_reduces() {
    let mut mat = Matrix {
        data: vec![vec![2.0, 1.0, 4.0], vec![1.0, 2.0, 5.0]],
        rows: 2,
        cols: 3,
    };

    let comp = Matrix {
        data: vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 2.0]],
        rows: 2,
        cols: 3,
    };

    rowreduce(&mut mat).unwrap();
    assert_eq!(mat.data, comp.data);
}

#[test]
fn it_upper_triangulates() {
    let mut mat = Matrix {
        data: vec![vec![1.0, 2.0, 1.0], vec![2.0, 1.0, 2.0], vec![3.0, 4.0, 6.0]],
        rows: 3,
        cols: 3,
    };

    let comp = Matrix {
        data: vec![vec![1.0, 2.0, 1.0], vec![0.0, -3.0, 0.0], vec![0.0, 0.0, 3.0]],
        rows: 3,
        cols: 3,

    };

    let res = upper_triangular(&mut mat).unwrap();

    assert_eq!(mat.data, comp.data);
}
