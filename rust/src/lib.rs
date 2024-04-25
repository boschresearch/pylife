
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use ndarray::prelude::*;


fn fourpoint_loop(turns: ArrayView1<f64>, turns_index: ArrayView1<usize>) -> (
    Array1<f64>, Array1<f64>, Array1<usize>, Array1<usize>, Array1<usize>
) {
    let len_turns = turns.len();
    let mut from_vals = Array1::<f64>::zeros(len_turns/2);
    let mut to_vals = Array1::<f64>::zeros(len_turns/2);
    let mut from_index = Array1::<usize>::zeros(len_turns/2);
    let mut to_index = Array1::<usize>::zeros(len_turns/2);

    let mut residual_index = Array1::<usize>::zeros(len_turns);

    let mut t_i: usize = 2;  // current turn index
    let mut r_i: usize = 2;  // current residual index
    let mut h_i: usize = 0;  // current hysteresis index

    residual_index[0] = 0;
    residual_index[1] = 1;

    while t_i < len_turns {
        if r_i < 3 {
            residual_index[r_i] = t_i;
            r_i += 1;
            t_i += 1;
            continue;
        }
        let a = turns[residual_index[r_i-3]];
        let b = turns[residual_index[r_i-2]];
        let c = turns[residual_index[r_i-1]];
        let d = turns[t_i];

        let ab = (a - b).abs();
        let bc = (b - c).abs();
        let cd = (c - d).abs();

        if bc <= ab && bc <= cd {
            from_vals[h_i] = b;
            to_vals[h_i] = c;

            r_i = r_i - 1;
            to_index[h_i] = turns_index[residual_index[r_i]];
            r_i = r_i - 1;
            from_index[h_i] = turns_index[residual_index[r_i]];
            h_i = h_i + 1;
            continue;
        }

        residual_index[r_i] = t_i;
        r_i = r_i + 1;
        t_i = t_i + 1;
    }
    (
        from_vals.slice(s![..h_i]).to_owned(),
        to_vals.slice(s![..h_i]).to_owned(),
        from_index.slice(s![..h_i]).to_owned(),
        to_index.slice(s![..h_i]).to_owned(),
        residual_index.slice(s![..r_i]).to_owned()
    )
}


#[pymodule]
fn _rust_lib<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {

    #[pyfn(m)]
    #[pyo3(name="fourpoint_loop")]
    fn fourpoint_loop_py<'py>(
        py: Python<'py>,
        turns: PyReadonlyArray1<f64>,
        turns_index: PyReadonlyArray1<usize>
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<usize>>,
    ) {
        let turns = turns.as_array();
        let turns_index = turns_index.as_array();
        let (
            from_vals, to_vals, from_index, to_index, residual_index
        ) = fourpoint_loop(turns, turns_index);
        (
            from_vals.into_pyarray_bound(py),
            to_vals.into_pyarray_bound(py),
            from_index.into_pyarray_bound(py),
            to_index.into_pyarray_bound(py),
            residual_index.into_pyarray_bound(py)
        )
    }
    Ok(())
}
