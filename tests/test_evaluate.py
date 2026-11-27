import numpy as np
import pandas as pd

from beta.evaluate import classification_problem_type


def test_many_integer_classes_are_multiclass_not_regression():
    rings = pd.Series(np.arange(4177) % 28 + 1)  # abalone: 28 integer ring counts
    assert classification_problem_type(rings) == "multiclass"


def test_two_classes_are_binary():
    assert classification_problem_type(pd.Series([0, 1, 1, 0])) == "binary"


def test_a_fit_far_over_its_time_limit_is_an_infrastructure_failure():
    import pytest
    from beta.evaluate import InfrastructureError, check_fit_duration

    check_fit_duration(330, 300)  # AutoGluon overruns a little; that is a normal fit
    with pytest.raises(InfrastructureError, match="suspended"):
        check_fit_duration(2 * 3600, 300)  # laptop slept for two hours mid-fit
