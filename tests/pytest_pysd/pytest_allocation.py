import pytest
import numpy as np
import xarray as xr
import portion as p

from pysd.py_backend.allocation import\
    Priorities, allocate_available,\
    allocate_by_priority, _allocate_by_priority_1d


class TestPriorities():
    @pytest.mark.parametrize(
        "q0,pp",
        [
            (  # rectangular
                np.array([6, 1, 3, 0, 0.5, 0]),
                np.array(
                    [[1, 5, 1, 0], [1, 10, 3, 0], [1, 4, 0.5, 0],
                     [1, 4, 0.5, 0], [1, 2, 1, 0], [1, 6, 3, 0]]
                )
            ),
            (  # triangular
                np.array([6, 1, 3, 0, 0.5, 0]),
                np.array(
                    [[2, 5, 1, 0], [2, 10, 3, 0], [2, 4, 0.5, 0],
                     [2, 4, 0.5, 0], [2, 2, 1, 0], [2, 6, 3, 0]]
                )
            ),
            (  # normal
                np.array([6, 1, 3, 0, 0.5, 0]),
                np.array(
                    [[3, 5, 1, 0], [3, 10, 3, 0], [3, 4, 0.5, 0],
                     [3, 4, 0.5, 0], [3, 2, 1, 0], [3, 6, 3, 0]]
                )
            ),
            (  # exponential
                np.array([6, 1, 3, 0, 0.5, 0]),
                np.array(
                    [[4, 5, 1, 0], [4, 10, 3, 0], [4, 4, 0.5, 0],
                     [4, 4, 0.5, 0], [4, 2, 1, 0], [4, 6, 3, 0]]
                )
            ),
        ],
        ids=["rectangular", "triangular", "normal", "exponential"]
    )
    def test_supply_demand(self, pp, q0):
        d_funcs, d_all, d_inter = Priorities.get_functions(q0, pp, "demand")

        pty = np.arange(0, 15, 0.05)
        for i in range(1, len(pty)):
            # assert the sum of functions
            assert d_all(pty[i]) == np.sum([d_f(pty[i]) for d_f in d_funcs])
            for d_f in d_funcs:
                # assert functions are monotonous
                assert d_f(pty[i-1]) >= d_f(pty[i])

            non_monotony = True
            for s_i, p_i, p_m in d_inter:
                if pty[i] in p_i or pty[i] == p_i.upper:
                    # assert functions are strictly monotonous in the interval
                    assert d_all(pty[i-1]) > d_all(pty[i])
                    non_monotony = False

            if non_monotony:
                # assert functions are constant out the interval
                assert d_all(pty[i-1]) == d_all(pty[i])

        for s_i, p_i, p_m in d_inter:
            assert p_m == .5 * (p_i.lower+p_i.upper)
            assert s_i.lower == d_all(p_i.upper)
            assert s_i.upper == d_all(p_i.lower)

    @pytest.mark.parametrize(
        "q0,pp",
        [
            (  # rectangular
                np.array([6, 1, 3, 0, 0.5, 0]),
                np.array(
                    [[1, 5, 1, 0], [1, 10, 3, 0], [1, 4, 0.5, 0],
                     [1, 4, 0.5, 0], [1, 2, 1, 0], [1, 6, 3, 0]]
                )
            ),
            (  # triangular
                np.array([6, 1, 3, 0, 0.5, 0]),
                np.array(
                    [[2, 5, 1, 0], [2, 10, 3, 0], [2, 4, 0.5, 0],
                     [2, 4, 0.5, 0], [2, 2, 1, 0], [2, 6, 3, 0]]
                )
            ),
            (  # normal
                np.array([6, 1, 3, 0, 0.5, 0]),
                np.array(
                    [[3, 5, 1, 0], [3, 10, 3, 0], [3, 4, 0.5, 0],
                     [3, 4, 0.5, 0], [3, 2, 1, 0], [3, 6, 3, 0]]
                )
            ),
            (  # exponential
                np.array([6, 1, 3, 0, 0.5, 0]),
                np.array(
                    [[4, 5, 1, 0], [4, 10, 3, 0], [4, 4, 0.5, 0],
                     [4, 4, 0.5, 0], [4, 2, 1, 0], [4, 6, 3, 0]]
                )
            ),
        ],
        ids=["rectangular", "triangular", "normal", "exponential"]
    )
    def test_supply_supply(self, pp, q0):
        error_message = "get_function_supply is not implemented"
        with pytest.raises(NotImplementedError, match=error_message):
            Priorities.get_functions(q0, pp, "supply")

    @pytest.mark.parametrize(
        "q0,pp,distance",
        [
            (  # rectangular
                np.array([6, 1, 3, 2, 0.5, 4]),
                np.array(
                    [[1, 5, 1, 0], [1, 10, 3, 0], [1, 4, 0.5, 0],
                     [1, 4, 0.5, 0], [1, 2, 1, 0], [1, 6, 3, 0]]
                ),
                0.5
            ),
            (  # triangular
                np.array([6, 1, 3, 3, 0.5, 4]),
                np.array(
                    [[2, 5, 1, 0], [2, 10, 3, 0], [2, 4, 0.5, 0],
                     [2, 4, 0.5, 0], [2, 2, 1, 0], [2, 6, 3, 0]]
                ),
                0.5
            ),
            (  # normal
                np.array([6, 1, 3, 3, 0.5, 4]),
                np.array(
                    [[3, 5, 1, 0], [3, 10, 3, 0], [3, 4, 0.5, 0],
                     [3, 4, 0.5, 0], [3, 2, 1, 0], [3, 6, 3, 0]]
                ),
                8.2923611
            ),
            (  # exponential
                np.array([6, 1, 3, 3, 0.5, 4]),
                np.array(
                    [[4, 5, 1, 0], [4, 10, 3, 0], [4, 4, 0.5, 0],
                     [4, 4, 0.5, 0], [4, 2, 1, 0], [4, 6, 3, 0]]
                ),
                36.7368005696
            ),
        ],
        ids=["rectangular", "triangular", "normal", "exponential"]
    )
    def test_priority_shape_demand(self, pp, q0, distance):
        for i in range(len(q0)):
            func, interval = Priorities.get_function_demand(q0[i], pp[i])
            xs = np.linspace(
                pp[i][1]-pp[i][2]*distance*0.95,
                pp[i][1]+pp[i][2]*distance*0.95,
                100
            )
            for j in range(1, len(xs)):
                # assert is monotically increasing
                assert func(xs[j-1]) > func(xs[j])

            xs_lower = np.linspace(
                pp[i][1]-3*pp[i][2]*distance,
                pp[i][1]-1.01*pp[i][2]*distance,
                100
            )
            for x in xs_lower:
                assert func(x) == q0[i]
            xs_upper = np.linspace(
                pp[i][1]+1.01*pp[i][2]*distance,
                pp[i][1]+3*pp[i][2]*distance,
                100
            )
            for x in xs_upper:
                assert np.isclose(func(x), 0, atol=1e-15)

            assert interval == p.open(
                pp[i][1]-pp[i][2]*distance,
                pp[i][1]+pp[i][2]*distance)

    @pytest.mark.parametrize(
        "pp,distance",
        [
            (  # rectangular
                np.array([1, 5, 1, 0]),
                0.5
            ),
            (  # triangular
                np.array([2, 5, 1, 0]),
                0.5
            ),
            (  # normal
                np.array([3, 5, 1, 0]),
                8.2923611
            ),
            (  # exponential
                np.array([4, 5, 1, 0]),
                36.7368005696
            ),
        ],
        ids=["rectangular", "triangular", "normal", "exponential"]
    )
    def test_priority_zeros_demand(self, pp, distance):
        func, interval = Priorities.get_function_demand(0, pp)
        xs = np.linspace(
            pp[1]-3*pp[2]*distance,
            pp[1]+3*pp[2]*distance,
            500
        )
        for x in xs:
            # assert is monotically increasing
            assert func(x) == 0

        assert interval == p.empty()

    @pytest.mark.parametrize(
        "pp,ptype,raise_type,error_message",
        [
            (  # invalid-kind
                np.array([[1, 10, 1, 0]]),
                "invalid",
                ValueError,
                r"kind='invalid' is not allowed\. kind should be "
                r"'demand' or 'supply'\."
            ),
            (  # fixed-quantity
                np.array([[0, 10, 1, 0]]),
                "demand",
                NotImplementedError,
                r"fixed_quantity priority profile is not implemented\."
            ),
            (  # constant-elasticity-demand
                np.array([[5, 10, 1, 0]]),
                "demand",
                NotImplementedError,
                r"Some results for Vensim showed some bugs when using "
                r"this priority curve\. Therefore, the curve is not "
                r"implemented in PySD as it cannot be properly tested\."
            ),
            (  # invalid-func
                np.array([[8, 10, 1, 0]]),
                "demand",
                ValueError,
                r"The priority function for pprofile=8 is not valid\."
            ),
            (  # supply
                np.array([[1, 10, 1, 0]]),
                "supply",
                NotImplementedError,
                r"get_function_supply is not implemented\."
            ),
            (  # negative-width
                np.array([[1, 10, -1, 0]]),
                "demand",
                ValueError,
                r"pwidth values must be positive\."
            ),
            (  # zero-width
                np.array([[1, 10, 0, 0]]),
                "demand",
                ValueError,
                r"pwidth values must be positive\."
            ),
        ],
        ids=["invalid-kind", "fixed-quantity", "constant-elasticity-demand",
             "invalid-func", "supply", "negative-width", "zero-width"]
    )
    def test_priorities_errors(self, pp, ptype, raise_type, error_message):
        with pytest.raises(raise_type, match=error_message):
            Priorities.get_functions(np.array([100]), pp, ptype)


class TestAllocateAvailable():

    @pytest.mark.parametrize(
        "requests,pp,avail,raise_type,error_message",
        [
            (  # negative-request
                xr.DataArray([6, -3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray(
                    [[1, 10, 1, 0], [1, 10, 1, 0], [1, 10, 1, 0]],
                    {'dim': ["A", "B", "C"],
                     'pprofile': ["ptype", "ppriority", "pwidth", "pextra"]}
                ),
                15,
                ValueError,
                r"There are some negative request values\. Ensure that "
                r"your request is always non-negative\. Allocation requires "
                r"all quantities to be positive or 0\.\n.*"
            ),
            (  # negative supply
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray(
                    [[1, 10, 1, 0], [1, 10, 1, 0], [1, 10, 1, 0]],
                    {'dim': ["A", "B", "C"],
                     'pprofile': ["ptype", "ppriority", "pwidth", "pextra"]}
                ),
                -7.5,
                ValueError,
                r"avail=-7\.5 is not allowed\. "
                r"avail should be non-negative\."
            ),
        ],
    )
    def test_allocate_available_errors(self, requests, pp, avail,
                                       raise_type, error_message):
        with pytest.raises(raise_type, match=error_message):
            allocate_available(requests, pp, avail)


class TestAllocateByPriority():
    @pytest.mark.parametrize(
        "requests,priority,width,supply,expected",
        [
            (
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray([10, 1, 0], {'dim': ["A", "B", "C"]}),
                3,
                15,
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
            ),
            (
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray([10, 1, 0], {'dim': ["A", "B", "C"]}),
                3,
                5,
                xr.DataArray([5, 0, 0], {'dim': ["A", "B", "C"]}),
            ),
            (
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray([10, 5, 0], {'dim': ["A", "B", "C"]}),
                3,
                7.5,
                xr.DataArray([6, 1.5, 0], {'dim': ["A", "B", "C"]}),
            ),
        ],
    )
    def test_allocate_by_priority(self, requests, priority, width,
                                  supply, expected):
        # Test some simple cases, the complicate cases are tested with
        # a full integration test
        assert allocate_by_priority(
            requests, priority, width, supply).equals(expected)

    @pytest.mark.parametrize(
        "requests,priority,width,supply,expected",
        [
            (
                np.array([6, 3, 3]),
                np.array([10, 1, 0]),
                3,
                15,
                np.array([6, 3, 3]),
            ),
            (
                np.array([6, 3, 3]),
                np.array([10, 1, 0]),
                3,
                5,
                np.array([5, 0, 0]),
            ),
            (
                np.array([6, 3, 3]),
                np.array([10, 5, 0]),
                3,
                7.5,
                np.array([6, 1.5, 0]),
            ),
        ],
    )
    def test__allocate_by_priority_1d(self, requests, priority, width,
                                      supply, expected):
        # Test some simple cases, the complicate cases are tested with
        # a full integration test
        assert np.all(_allocate_by_priority_1d(
            requests, priority, width, supply) == expected)

    @pytest.mark.parametrize(
        "requests,priority,width,supply,raise_type,error_message",
        [
            (  # negative-request
                xr.DataArray([6, -3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray([10, 1, 0], {'dim': ["A", "B", "C"]}),
                3,
                15,
                ValueError,
                r"There are some negative request values\. Ensure that "
                r"your request is always non-negative\. Allocation requires "
                r"all quantities to be positive or 0\.\n.*"
            ),
            (  # 0 width
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray([10, 1, 0], {'dim': ["A", "B", "C"]}),
                0,
                5,
                ValueError,
                r"width=0 is not allowed\. width should be greater than 0\."
            ),
            (  # negative width
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray([10, 5, 0], {'dim': ["A", "B", "C"]}),
                -3,
                7.5,
                ValueError,
                r"width=-3 is not allowed\. width should be greater than 0\."
            ),
            (  # negative supply
                xr.DataArray([6, 3, 3], {'dim': ["A", "B", "C"]}),
                xr.DataArray([10, 5, 0], {'dim': ["A", "B", "C"]}),
                3,
                -7.5,
                ValueError,
                r"supply=-7\.5 is not allowed\. "
                r"supply should be non-negative\."
            ),
        ],
    )
    def test_allocate_by_priority_errors(self, requests, priority,
                                         width, supply, raise_type,
                                         error_message):
        with pytest.raises(raise_type, match=error_message):
            allocate_by_priority(requests, priority, width, supply)
