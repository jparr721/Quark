use std::fmt;

/// The Matrix Struct is a size-aware 2d vector
#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    pub nrows: usize,
    pub ncols: usize,
}

impl Matrix {
    pub fn new(data: Vec<Vec<f64>>) -> Matrix {
        let rows = data.len();
        let cols = data[0].len();
        Matrix {
            data: data,
            nrows: rows,
            ncols: cols,
        }
    }

    /// Merges one matrix into another one vec by vec
    pub fn merge(
        &mut self,
        other: &mut Matrix,
        inplace: Option<bool>,
    ) -> Result<Option<Matrix>, &'static str> {
        // This is highly inefficient, need to work on this later...
        let internal_clone = self.clone();
        let other_clone = other.clone();

        if other_clone.shape() != internal_clone.shape() {
            return Err("Shape misaslignment");
        }

        if inplace.unwrap_or(true) {
            for i in 0..self.data.len() {
                self.data[i].append(&mut other.data[i]);
            }
            return Ok(None);
        }

        let mut clone = self.data.clone();

        for i in 0..clone.len() {
            clone[i].append(&mut other.data[i]);
        }

        Ok(Some(Matrix::new(clone)))
    }

    pub fn shape(self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    pub fn mean(self) -> Result<Option<Matrix>, &'static str> {
        if self.nrows < 1 {
            return Err("Matrix height must be greater than 0");
        }

        let mut out = Matrix::new(vec![Vec::with_capacity(1); self.ncols]);

        for i in 0..self.ncols {
            let mut col = vec![0.0; self.nrows];
            for j in 0..self.nrows {
                col.push(self.data[j][i]);
            }

            let mean = col.iter().fold(0.0, |a, &b| a + b) / col.len() as f64;
            out.data[0].push(mean);
        }

        Ok(Some(out))
    }

    pub fn t(&mut self, inplace: bool) -> Result<Option<Matrix>, &'static str> {
        if self.nrows < 1 || self.ncols < 1 {
            return Err("Data must be present to transpose");
        }

        let mut t = vec![Vec::with_capacity(self.nrows); self.ncols];

        for r in &mut self.data {
            for i in 0..self.nrows {
                t[i].push(r[i]);
            }
        }

        if inplace {
            self.data = t;
            Ok(None)
        } else {
            Ok(Some(Matrix::new(t)))
        }
    }

    pub fn swap(
        &mut self,
        p: usize,
        q: usize,
        inplace: Option<bool>,
    ) -> Result<Option<Matrix>, &'static str> {
        if self.nrows < 2 {
            return Err("There must be at least 2 rows in order to swap");
        }

        if p > self.nrows || q > self.nrows {
            return Err("You cannot specify a p less than the matrix height");
        }

        if p == q {
            if inplace.unwrap_or(true) {
                return Ok(None);
            }
        }

        self.data.swap(p, q);

        if inplace.unwrap_or(true) {
            return Ok(Some(self.clone()));
        }

        Ok(None)
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
    if first.ncols != second.nrows {
        return Err("Row-Column mis-alignment");
    }

    let mut vec = vec![vec![0.0; first.nrows]; second.ncols];

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
    let mat = Matrix::new(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]);

    assert_eq!(mat.shape(), (2, 3));
}

#[test]
fn it_makes_matrix() {
    let data = vec![vec![0.0; 2]; 2];
    let mat = Matrix::new(data.clone());

    assert_eq!(data, mat.data);
}

#[test]
fn it_transposes_matrix() {
    let mut data = Matrix {
        data: vec![vec![0.0, 0.0], vec![1.0, 1.0]],
        nrows: 2,
        ncols: 2,
    };

    let comp = Matrix {
        data: vec![vec![0.0, 1.0], vec![0.0, 1.0]],
        nrows: 2,
        ncols: 2,
    };

    data.t(true).unwrap();

    assert_eq!(comp.data, data.data);
}

#[test]
fn it_matrix_multiplies() {
    let data = Matrix {
        data: vec![vec![1.0, 2.0], vec![2.0, 1.0]],
        nrows: 2,
        ncols: 2,
    };

    let data2 = Matrix {
        data: vec![vec![2.0, 1.0], vec![1.0, 2.0]],
        nrows: 2,
        ncols: 2,
    };

    let comp = Matrix {
        data: vec![vec![4.0, 5.0], vec![5.0, 4.0]],
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
    let mat = Matrix::new(vec![vec![0.0; 3], vec![0.0; 3]]);
    let comp = Matrix::new(vec![vec![0.0; 3], vec![0.0; 3]]);

    assert_eq!(mat.data, comp.data);
}
