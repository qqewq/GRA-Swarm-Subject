"""
Базовые тесты для класса Agent.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.agent import Agent

def test_agent_creation():
    ag = Agent(dim=10)
    assert ag.dim == 10
    assert ag.phi == 0.8
    print("test_agent_creation пройден")

def test_subject_setting():
    ag = Agent(dim=10)
    ag.set_as_initial_subject()
    assert ag.is_subject()
    assert ag.S == 1.0
    assert ag.phi == 0.0
    print("test_subject_setting пройден")

if __name__ == "__main__":
    test_agent_creation()
    test_subject_setting()
