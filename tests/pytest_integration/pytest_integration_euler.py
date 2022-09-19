from pathlib import Path

import pytest
import numpy as np
import pandas as pd


def harmonic_position(t, x0, k, m):
    """
    Position for the simple harmonic oscillator
    'test-models/samples/simple_harmonic_oscillator
     /simple_harmonic_oscillator.mdl'
    """
    return x0*np.cos(np.sqrt(k/m)*t)


def harmonic_speed(t, x0, k, m):
    """
    Speed for the simple harmonic oscillator
    'test-models/samples/simple_harmonic_oscillator
     /simple_harmonic_oscillator.mdl'
    """
    return - x0*np.sqrt(k/m)*np.sin(np.sqrt(k/m)*t)


@pytest.mark.parametrize(
    "model_path,f,f2,stocks,arguments,integration_frame,ts_log,ts_rmse",
    [
        # model_path: model path: pathlib.Path object
        # f: stocks analytical solutions: tuple of funcs
        # f2: stocks analytical solutions second derivative: tuple of funcs
        # stocks: stock names in the model: tuple of strings
        # arguments: arguments of fs and f2s in the model: tuple of strings
        # integration_frame: minimum and maximum time to find solutions: tuple
        # ts_log: logarithmic range of time steps in base 10: tuple
        # ts_rmse: sorted time step for RMSE test: iterable
        (
            Path("test-models/samples/teacup/teacup.mdl"),
            (lambda t, T0, TR, ct: TR + (T0-TR)*np.exp(-t/ct),),
            (lambda t, T0, TR, ct: (T0-TR)*np.exp(-t/ct)/ct**2,),
            ("Teacup Temperature",),
            ("Teacup Temperature", "Room Temperature", "Characteristic Time"),
            (0, 40),
            (1, -5),
            [10, 5, 1, 0.5, 0.1, 0.05, 0.01]
        ),
        (
            Path("test-models/samples/simple_harmonic_oscillator/"
                 "simple_harmonic_oscillator.mdl"),
            (
                harmonic_position,
                harmonic_speed
            ),
            (
                lambda t, x0, k, m: -k/m*harmonic_position(t, x0, k, m),
                lambda t, x0, k, m: -k/m*harmonic_speed(t, x0, k, m)
            ),
            ("position", "speed"),
            ("initial position", "elastic constant", "mass"),
            (0, 40),
            (-1, -5),
            [10, 5, 1, 0.5, 0.1, 0.05, 0.01]
        )
    ],
    ids=["teacup", "harmonic"]
)
class TestEulerConvergence:
    """
    Tests for Euler integration method convergence.
    """
    # Number of points to compute the tests
    n_points_lte = 30

    def test_local_truncation_error(self, model, f, f2, stocks, arguments,
                                    integration_frame, ts_log, ts_rmse):
        """
        Test the local truncation error (LTE).
        LTE = y_1 - y(t_0+h) = 0.5*h**2*y''(x) for x in [t_0, t_0+h]

        where y_1 = y(t_0) + h*f(t_0, y(t_0)) and

        Generates n_points_lte in the given integration frame and test the
        convergence with logarithmically uniform split time_steps.

        Parameter
        ---------
        model: pysd.py_backend.model.Model
            The model to integrate.
        f: tuple of functions
            The functions of the analytical solution of each stock.
        f2: tuple of functions
            The second derivative of the functions of the analytical
            solution of each stock.
        stocks: tuple of strings
            The name of the stocks.
        arguments: tuple of strings
            The neccessary argument names to evaluate f's and f2's.
            Note that all the functions must take the same arguments
            and in the same order.
        integration_frame: tuple
            Initial time of the model (usually 0) and maximum time to
            generate a value for test the LTE.
        ts_log: tuple
            log in base 10 of the inteval of time step to generate. I.e.,
            the first point will be evaluated with time_step = 10**ts_log[0]
            and the last one with 10**ts_log[1].
        ts_rmse: iterable
            Not used.

        """
        # Generate starting points to compute LTE
        t0s = np.random.uniform(*integration_frame, self.n_points_lte)
        # Generate time steps
        hs = 10**np.linspace(*ts_log, self.n_points_lte)
        # Get model values before making any change
        model_values = [model[var] for var in arguments]

        for t0, h in zip(t0s, hs):
            # Reload model
            model.reload()
            # Get start value(s)
            x0s = [
                func(t0, *model_values)
                for func in f
            ]
            # Get expected value(s)
            x_expect = np.array([
                func(t0 + h, *model_values)
                for func in f
            ])
            # Get error bound (error = 0.5h²*f''(x) for x in [t0, t0+h])
            # The 0.5 factor is removed to avoid problems with local maximums
            # We assume error < h²*max(f''(x)) for x in  [t0, t0+h]
            error = h**2*np.array([
                max(func(np.linspace(t0, t0+h, 1000), *model_values))
                for func in f2
            ])
            # Run the model from (t0, x0s) to t0+h
            ic = t0, {stock: x0 for stock, x0 in zip(stocks, x0s)}
            x_euler = model.run(
                initial_condition=ic,
                time_step=h,
                return_columns=stocks,
                return_timestamps=t0+h
            ).values[0]

            # Expected error
            assert np.all(np.abs(x_expect - x_euler) <= np.abs(error)),\
                f"The LTE is bigger than the expected one ({ic}) h={h},"\
                f"\n{np.abs(x_expect - x_euler)} !<=  {np.abs(error)}, "

    def test_root_mean_square_error(self, model, f, f2, stocks, arguments,
                                    integration_frame, ts_log, ts_rmse):
        """
        Test the root-mean-square error (RMSE).
        RMSE = SQRT(MEAN(y_i-y(t_0+h*i)))

        Integrates the given model with different time steps and checks
        that the RMSE decreases when the time step decreases.

        Parameter
        ---------
        model: pysd.py_backend.model.Model
            The model to integrate.
        f: tuple of functions
            The functions of the analytical solution of each stock.
        f2: tuple of functions
            Not used.
        stocks: tuple of strings
            The name of the stocks.
        arguments: tuple of strings
            The neccessary argument names to evaluate f's and f2's.
            Note that all the functions must take the same arguments
            and in the same order.
        integration_frame: tuple
            Not used.
        ts_log: tuple
            Not used.
        ts_rmse: iterable
            Time step to compute the root mean square error over the
            whole integration. It shopuld be sorted from biggest to
            smallest.

        """
        # Get model values before making any change
        model_values = [model[var] for var in arguments]

        rmse = []
        for h in ts_rmse:
            # Reload model
            model.reload()
            # Run the model from (t0, x0s) to t0+h
            x_euler = model.run(
                time_step=h,
                saveper=h,
                return_columns=stocks
            )
            # Expected values
            expected_values = pd.DataFrame(
                index=x_euler.index,
                data={
                    stock: func(x_euler.index, *model_values)
                    for stock, func in zip(stocks, f)
                }
            )
            # Compute the RMSE for each stock
            rmse.append(np.sqrt(((x_euler-expected_values)**2).mean()))

        # Assert that the RMSE decreases for all stocks while
        # decreasing the time step
        assert np.all(np.diff(rmse, axis=0) < 0)
