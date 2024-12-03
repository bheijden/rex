import pytest
import tempfile
import numpy as onp
import rex.gmm_estimator as gmm_estimator


@pytest.mark.parametrize("is_deterministic", [True, False])
def test_estimator(is_deterministic):
    # Create array of 20 data points
    if is_deterministic:
        data = onp.ones(20)
    else:  # Else, create random non-negative data
        data = onp.clip(onp.random.randn(20) + 2, 0, None)

    # Create estimator
    estimator = gmm_estimator.GMMEstimator(data)

    # Test fit (only 10 steps)
    estimator.fit(num_steps=10)

    # Test get dist
    _dist = estimator.get_dist()
    _dist = estimator.get_dist(percentile=0.1)

    # Test plotting
    _ax = estimator.plot_loss()
    _ax = estimator.plot_normalized_weights()
    _ax = estimator.plot_hist()

    # Test animation
    if not is_deterministic:
        ani = estimator.animate_training(num_frames=4, xmin=0, xmax=4, num_points=1000)

        # Save animation to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=True) as temp_file:
            ani.save(temp_file.name, writer="pillow", fps=30)
