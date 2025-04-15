from epioncho_ibm.state.people import DelayArrays
from epioncho_ibm.state.params import Params, BlackflyParams
import numpy as np

def test_delay_upscale_even():
    worm_delay=np.array([
        [1,2,3],
        [4,5,6]
    ])
    _, new_worm_delay = DelayArrays.upscale_delay_arrays(
        n_rows_new=4,
        old_delay_array=worm_delay,
        n_cols=worm_delay.shape[1],
        current_index=0
    )
    np.testing.assert_array_equal(
        new_worm_delay,
        np.array([
            [1, 2, 3],
            [0, 0, 0],
            [4, 5, 6],
            [0, 0, 0]
        ])
    )

def test_delay_upscale_odd():
    worm_delay=np.array([
        [1,2,3],
        [4,5,6]
    ])
    _, new_worm_delay = DelayArrays.upscale_delay_arrays(
        n_rows_new=3,
        old_delay_array=worm_delay,
        n_cols=worm_delay.shape[1],
        current_index=0
    )
    np.testing.assert_array_equal(
        new_worm_delay,
        np.array([
            [1, 2, 3],
            [4, 5, 6],
            [0, 0, 0]
        ])
    )

def test_delay_downscale_even():
    worm_delay=np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9],
        [10,11,12]
    ])
    _, new_worm_delay = DelayArrays.downscale_delay_arrays(
        n_rows_new=2,
        old_delay_array=worm_delay,
        n_cols=worm_delay.shape[1],
        current_index=0
    )
    np.testing.assert_array_equal(
        new_worm_delay,
        np.array([
            [5,7,9],
            [17,19,21]
        ])
    )

def test_delay_downscale_odd():
    worm_delay=np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9],
        [10,11,12]
    ])
    _, new_worm_delay = DelayArrays.downscale_delay_arrays(
        n_rows_new=3,
        old_delay_array=worm_delay,
        n_cols=worm_delay.shape[1],
        current_index=0
    )
    np.testing.assert_array_equal(
        new_worm_delay,
        np.array([
            [5,7,9],
            [17,19,21],
            [0,0,0]
        ])
    )

def test_rescale_delay_arrays_upscale():
    individual_exposure = np.zeros(10)
    delay_arrays = DelayArrays.from_params(Params(
        n_people=10,
        delta_time_days=1,
        month_length_days=28,
        blackfly=BlackflyParams(
            l3_delay=10, # months
            l1_delay=4 # days
        )
    ), individual_exposure)

    delay_arrays._worm_delay_current = 7
    delay_arrays._mf_delay_current = 2
    delay_arrays._exposure_delay_current = 3

    delay_arrays.rescale_delay_arrays(Params(
        n_people=10,
        delta_time_days=0.5,
        month_length_days=28,
        blackfly=BlackflyParams(
            l3_delay=10, # months
            l1_delay=4 # days
        )
    ))
    assert delay_arrays._worm_delay.shape == (10*28/0.5, 10)
    assert delay_arrays._worm_delay_current == 14
    assert delay_arrays._mf_delay.shape == (4/0.5, 10)
    assert delay_arrays._mf_delay_current == 4
    assert delay_arrays._exposure_delay.shape == (4/0.5, 10)
    assert delay_arrays._exposure_delay_current == 6

def test_rescale_delay_arrays_downscale():
    individual_exposure = np.zeros(10)
    delay_arrays = DelayArrays.from_params(Params(
        n_people=10,
        delta_time_days=1,
        month_length_days=28,
        blackfly=BlackflyParams(
            l3_delay=10, # months
            l1_delay=4 # days
        )
    ), individual_exposure)

    delay_arrays._worm_delay_current = 7
    delay_arrays._mf_delay_current = 2
    delay_arrays._exposure_delay_current = 3
    
    delay_arrays.rescale_delay_arrays(Params(
        n_people=10,
        delta_time_days=7,
        month_length_days=28,
        blackfly=BlackflyParams(
            l3_delay=10, # months
            l1_delay=4 # days
        )
    ))
    assert delay_arrays._worm_delay.shape == (np.ceil(10*28/7), 10)
    assert delay_arrays._worm_delay_current == 1
    assert delay_arrays._mf_delay.shape ==  (np.ceil(4*1/7), 10)
    assert delay_arrays._mf_delay_current == 0
    assert delay_arrays._exposure_delay.shape ==  (np.ceil(4*1/7), 10)
    assert delay_arrays._exposure_delay_current == 0

def test_rescale_delay_arrays_no_change():
    individual_exposure = np.zeros(10)
    delay_arrays = DelayArrays.from_params(Params(
        n_people=10,
        delta_time_days=1,
        month_length_days=28,
        blackfly=BlackflyParams(
            l3_delay=10, # months
            l1_delay=4 # days
        )
    ), individual_exposure)

    delay_arrays._worm_delay_current = 7
    delay_arrays._mf_delay_current = 2
    delay_arrays._exposure_delay_current = 3
    
    delay_arrays.rescale_delay_arrays(Params(
        n_people=10,
        delta_time_days=1,
        month_length_days=28,
        blackfly=BlackflyParams(
            l3_delay=10, # months
            l1_delay=4 # days
        )
    ))
    
    assert delay_arrays._worm_delay.shape == (np.ceil(10*28/1), 10)
    assert delay_arrays._worm_delay_current == 7
    assert delay_arrays._mf_delay.shape ==  (np.ceil(4*1/1), 10)
    assert delay_arrays._mf_delay_current == 2
    assert delay_arrays._exposure_delay.shape ==  (np.ceil(4*1/1), 10)
    assert delay_arrays._exposure_delay_current == 3
