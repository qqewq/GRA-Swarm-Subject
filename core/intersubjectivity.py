import numpy as np
from .agent import Agent

def cross_foam(agent_i: Agent, agent_j: Agent) -> float:
    """
    Интерсубъективная пена Φ_cross между двумя агентами.
    Мера расхождения их моделей мира.
    """
    return np.linalg.norm(agent_i.M - agent_j.M) ** 2

def intersubjective_step(agent_i: Agent, agent_j: Agent, lr: float = 0.01) -> None:
    """
    Один шаг интерсубъективного обнуления для пары агентов.
    Оба агента подстраивают свои модели, чтобы уменьшить перекрёстную пену.
    """
    # Градиент cross_foam по M_i: 2*(M_i - M_j)
    grad_i = 2 * (agent_i.M - agent_j.M)
    grad_j = 2 * (agent_j.M - agent_i.M)

    agent_i.M -= lr * grad_i
    agent_j.M -= lr * grad_j

    # Нормировка
    for ag in (agent_i, agent_j):
        norm = np.linalg.norm(ag.M)
        if norm > 1e-8:
            ag.M /= norm
