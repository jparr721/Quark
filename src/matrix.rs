use std::fmt;

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

pub fn matmul(first: Matrix, second: Matrix) -> Result<Matrix, &'static str> {
    if first.cols != second.rows {
        return Err("Row-Column mis-alignment");
    }

    let mut vec = vec![vec![0.0; first.rows]; second.cols];

    for i in 0..first.rows {
        for j in 0..second.cols {
            for k in 0..second.rows {
                vec[i][j] = vec[i][j] + first.data[i][k] * second.data[k][j];
            }
        }
    }

    Ok(Matrix {
        data: vec,
        rows: first.rows,
        cols: second.cols,
    })
}

pub fn row_replacement(matrix: &mut Matrix, i: usize, j: usize, factor: f64) {
    for k in 0..matrix.data[j].len() {
        matrix.data[j][k] += matrix.data[i][k] * factor;
    }
}

pub fn scale(matrix: &mut Matrix, row: usize, factor: f64) {
    let n = matrix.cols;

    for i in 0..n {
        matrix.data[row][i] *= factor;
    }
}

pub fn rowreduce(mat: &mut Matrix) -> Result<Matrix, &'static str> {
    let n = mat.clone().rows();

    for i in 0..n {
        // Find a pivot point
        for j in i..n {
            if mat.data[j][i] != 0.0 {
                if i != j {
                    mat.swap(i, j);
                }
                break;
            }

            if j == n - 1 {
                return Err("No pivots found in matrix!");
            }
        }

        for j in i + 1..n {
            row_replacement(mat, i, j, -mat.data[j][i] / mat.data[i][i]);
        }
    }

    // Back subsitution
    for i in (0..n).rev() {
        for j in 0..i {
            row_replacement(mat, i, j, -mat.data[j][i] / mat.data[i][i]);
        }
    }

    // Ones along diagonal
    for i in 0..n {
        scale(mat, i, 1.0 / mat.data[i][i]);
    }

    let matrix = Matrix {
        rows: mat.clone().len(),
        cols: mat.data[0].len(),
        data: mat.data.clone(),
    };

    Ok(matrix)
}

/// Performs a reduction operation on a given matrix via gauss-jordan elimination
pub fn gaussj(mat: &mut Matrix, rhs: &mut Matrix) -> Result<(Matrix, Matrix), &'static str> {
    let n = mat.data.len();
    let mut irow = 0;
    let mut icol = 0;
    let mut pivinv = 0.0;

    let indxc = &mut vec![1; n];
    let indxr = &mut vec![1; n];
    let ipiv = &mut vec![0.0; n];

    for i in 0..n {
        let mut big = 0.0;
        // Find a pivot point
        for j in 0..n {
            if ipiv[j] != 1.0 {
                for k in 0..n {
                    if mat.data[j][k].abs() >= big {
                        big = mat.data[j][k].abs();
                        irow = j;
                        icol = k;
                    }
                }
            } else if ipiv[n - 1] > 1.0 {
                return Err("Singular matrix");
            }
        }
    }
    ipiv[icol] = ipiv[icol] + 1.0;

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
    if mat.data[icol][icol] == 0.0 {
        return Err("Singular matrix");
    }

    // Take our pivot position
    pivinv = 1.0 / mat.data[icol][icol];

    // Now, set the pivot spot to one
    mat.data[icol][icol] = 1.0;

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

    let res = rowreduce(&mut mat).unwrap();
    assert_eq!(res.data, comp.data);
}
