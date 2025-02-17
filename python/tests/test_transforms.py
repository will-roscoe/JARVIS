import pytest
import numpy as np
from jarvis.transforms import gradmap, dropthreshold, coadd, normalize, align_cmls
from jarvis.const import FITSINDEX

class TestGradmap:
    def test_gradmap_basic(self):
        input_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_output = np.array([[10.63014581,  8.94427191, 10.63014581],
                                    [16.97056275, 14.14213562, 16.97056275],
                                    [10.63014581,  8.94427191, 10.63014581]])
        output = gradmap(input_arr)
        np.testing.assert_almost_equal(output, expected_output, decimal=5)

    def test_gradmap_zero_input(self):
        input_arr = np.zeros((3, 3))
        expected_output = np.zeros((3, 3))
        output = gradmap(input_arr)
        np.testing.assert_array_equal(output, expected_output)

    def test_gradmap_custom_kernel(self):
        input_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        custom_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        expected_output = np.array([[ 6.,  6.,  6.],
                                    [ 6.,  6.,  6.],
                                    [ 6.,  6.,  6.]])
        output = gradmap(input_arr, kernel2d=custom_kernel)
        np.testing.assert_almost_equal(output, expected_output, decimal=5)

    def test_gradmap_different_boundary(self):
        input_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_output = np.array([[10.63014581,  8.94427191, 10.63014581],
                                    [16.97056275, 14.14213562, 16.97056275],
                                    [10.63014581,  8.94427191, 10.63014581]])
        output = gradmap(input_arr, boundary='fill')
        np.testing.assert_almost_equal(output, expected_output, decimal=5)

    def test_gradmap_different_mode(self):
        input_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_output = np.array([[ 7.07106781,  7.07106781,  7.07106781],
                                    [ 7.07106781,  7.07106781,  7.07106781],
                                    [ 7.07106781,  7.07106781,  7.07106781]])
        output = gradmap(input_arr, mode='valid')
        np.testing.assert_almost_equal(output, expected_output, decimal=5)

class TestDropthreshold:
    def test_dropthreshold_basic(self):
        input_arr = np.array([1, 2, 3, 4, 5])
        threshold = 3
        expected_output = np.array([0, 0, 3, 4, 5])
        output = dropthreshold(input_arr, threshold)
        np.testing.assert_array_equal(output, expected_output)

    def test_dropthreshold_all_below_threshold(self):
        input_arr = np.array([1, 2, 3, 4, 5])
        threshold = 6
        expected_output = np.array([0, 0, 0, 0, 0])
        output = dropthreshold(input_arr, threshold)
        np.testing.assert_array_equal(output, expected_output)

    def test_dropthreshold_all_above_threshold(self):
        input_arr = np.array([1, 2, 3, 4, 5])
        threshold = 0
        expected_output = np.array([1, 2, 3, 4, 5])
        output = dropthreshold(input_arr, threshold)
        np.testing.assert_array_equal(output, expected_output)

    def test_dropthreshold_with_negative_values(self):
        input_arr = np.array([-1, -2, 0, 1, 2])
        threshold = 0
        expected_output = np.array([0, 0, 0, 1, 2])
        output = dropthreshold(input_arr, threshold)
        np.testing.assert_array_equal(output, expected_output)

    def test_dropthreshold_with_floats(self):
        input_arr = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        threshold = 3.5
        expected_output = np.array([0, 0, 3.5, 4.5, 5.5])
        output = dropthreshold(input_arr, threshold)
        np.testing.assert_array_equal(output, expected_output)
class TestCoadd:
    def test_coadd_basic(self):
        input_arrs = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        expected_output = np.array([4, 5, 6])
        output = coadd(input_arrs)
        np.testing.assert_array_equal(output, expected_output)

    def test_coadd_with_weights(self):
        input_arrs = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        weights = [0.1, 0.3, 0.6]
        expected_output = np.array([5.2, 6.2, 7.2])
        output = coadd(input_arrs, weights)
        np.testing.assert_array_equal(output, expected_output)

    def test_coadd_single_array(self):
        input_arrs = [np.array([1, 2, 3])]
        expected_output = np.array([1, 2, 3])
        output = coadd(input_arrs)
        np.testing.assert_array_equal(output, expected_output)

    def test_coadd_empty_list(self):
        input_arrs = []
        with pytest.raises(ValueError):
            coadd(input_arrs)

    def test_coadd_different_shapes(self):
        input_arrs = [np.array([1, 2]), np.array([3, 4, 5])]
        with pytest.raises(ValueError):
            coadd(input_arrs)
class TestNormalize:
    def test_normalize_basic(self):
        input_arr = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([0, 0.25, 0.5, 0.75, 1])
        output = normalize(input_arr)
        np.testing.assert_almost_equal(output, expected_output, decimal=5)

    def test_normalize_with_negative_values(self):
        input_arr = np.array([-1, 0, 1, 2, 3])
        expected_output = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
        output = normalize(input_arr)
        np.testing.assert_almost_equal(output, expected_output, decimal=5)

    def test_normalize_single_value(self):
        input_arr = np.array([5])
        expected_output = np.array([0])
        output = normalize(input_arr)
        np.testing.assert_array_equal(output, expected_output)

    def test_normalize_all_same_value(self):
        input_arr = np.array([3, 3, 3, 3, 3])
        expected_output = np.array([0, 0, 0, 0, 0])
        output = normalize(input_arr)
        np.testing.assert_array_equal(output, expected_output)

    def test_normalize_with_floats(self):
        input_arr = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        expected_output = np.array([0, 0.25, 0.5, 0.75, 1])
        output = normalize(input_arr)
        np.testing.assert_almost_equal(output, expected_output, decimal=5)
class TestAlignCmls:
    def test_align_cmls_basic(self, mocker):
        # Mocking fits and headers
        mock_fits = mocker.MagicMock()
        mock_fits[FITSINDEX].data.shape = (3, 3)
        mock_fits[FITSINDEX].header = {'CML': 0}
        
        input_fits = [mock_fits, mock_fits, mock_fits]
        primary_index = 0
        
        mocker.patch('jarvis.transforms.fitsheader', return_value=0)
        mocker.patch('jarvis.transforms.fits_from_parent', return_value=mock_fits)
        
        output = align_cmls(input_fits, primary_index)
        
        assert len(output) == len(input_fits)
        for out in output:
            assert out == mock_fits

    def test_align_cmls_different_cml(self, mocker):
        # Mocking fits and headers
        mock_fits1 = mocker.MagicMock()
        mock_fits1[FITSINDEX].data.shape = (3, 3)
        mock_fits1[FITSINDEX].header = {'CML': 0}
        
        mock_fits2 = mocker.MagicMock()
        mock_fits2[FITSINDEX].data.shape = (3, 3)
        mock_fits2[FITSINDEX].header = {'CML': 90}
        
        input_fits = [mock_fits1, mock_fits2]
        primary_index = 0
        
        mocker.patch('jarvis.transforms.fitsheader', side_effect=[0, 90])
        mocker.patch('jarvis.transforms.fits_from_parent', return_value=mock_fits1)
        
        output = align_cmls(input_fits, primary_index)
        
        assert len(output) == len(input_fits)
        for out in output:
            assert out == mock_fits1

    def test_align_cmls_shape_mismatch(self,mocker):
        # Mocking fits and headers
        mock_fits1 = mocker.MagicMock()
        mock_fits1[FITSINDEX].data.shape = (3, 3)
        mock_fits1[FITSINDEX].header = {'CML': 0}
        
        mock_fits2 = mocker.MagicMock()
        mock_fits2[FITSINDEX].data.shape = (4, 4)
        mock_fits2[FITSINDEX].header = {'CML': 90}
        
        input_fits = [mock_fits1, mock_fits2]
        primary_index = 0
        
        with pytest.raises(AssertionError):
            align_cmls(input_fits, primary_index)





