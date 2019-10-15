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
}
