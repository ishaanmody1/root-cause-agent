import pytest

from rootcause.detect.detector import detect
from rootcause.sim.scenarios import make_scenario
from rootcause.sim.simulator import simulate


@pytest.fixture
def clog_scenario():
    sc = simulate(make_scenario("clog", 6))
    return sc, detect(sc.data)
