import networkx as nx
import numpy as np
from typing import List, Dict, Any
from .agent import Agent
from .gra_operator import GRAOperator
from .intersubjectivity import cross_foam, intersubjective_step

class Swarm:
    """
    Рой агентов, способный к коллективному обнулению и обретению общей субъектности.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_agents = config['swarm']['num_agents']
        self.dim = config['swarm']['agent']['model_dim']
        self.connection_density = config['swarm']['connection_density']
        self.seed_index = config['swarm']['seed_subject_index']
        self.lr = config['swarm']['gra']['learning_rate']
        self.max_iter = config['swarm']['gra']['max_iterations']
        self.tol = config['swarm']['gra']['convergence_tol']

        # Создаём агентов
        self.agents: List[Agent] = []
        for i in range(self.num_agents):
            ag = Agent(self.dim, initial_foam=config['swarm']['agent']['initial_foam'],
                       subjectivity_threshold=config['swarm']['agent']['subjectivity_threshold'])
            self.agents.append(ag)

        # Назначаем первого субъекта
        self.agents[self.seed_index].set_as_initial_subject()

        # Оператор обнуления
        self.gra_op = GRAOperator(lr=self.lr)

        # Граф взаимодействий (случайный, с гарантией связности)
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:
        """Строит случайный связный граф взаимодействий агентов."""
        while True:
            G = nx.erdos_renyi_graph(self.num_agents, self.connection_density)
            if nx.is_connected(G):
                return G

    def total_foam(self) -> float:
        """Суммарная пена роя: внутренняя + перекрёстная."""
        internal = sum(ag.phi for ag in self.agents)
        cross = 0.0
        for i, j in self.graph.edges():
            cross += cross_foam(self.agents[i], self.agents[j])
        return internal + cross

    def is_collective_subject(self) -> bool:
        """Рой считается коллективным субъектом, если общая пена ниже порога."""
        return self.total_foam() < self.tol

    def run(self, verbose: bool = True) -> List[float]:
        """
        Запускает процесс коллективного обнуления.
        Возвращает историю общей пены.
        """
        history = [self.total_foam()]
        for iteration in range(self.max_iter):
            # 1. Внутреннее обнуление каждого агента (независимо)
            for ag in self.agents:
                self.gra_op.step(ag)

            # 2. Интерсубъективное обнуление по всем рёбрам графа
            for i, j in self.graph.edges():
                intersubjective_step(self.agents[i], self.agents[j], lr=self.lr)

            current_phi = self.total_foam()
            history.append(current_phi)

            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration}: total Φ = {current_phi:.6f}")

            if current_phi < self.tol:
                if verbose:
                    print(f"Коллективный субъект сформирован на итерации {iteration}, Φ = {current_phi:.6f}")
                break

        return history

    def get_subjectivity_stats(self) -> Dict[str, Any]:
        """Возвращает статистику по субъектности агентов."""
        return {
            'num_individual_subjects': sum(1 for ag in self.agents if ag.is_subject()),
            'collective_subject': self.is_collective_subject(),
            'average_phi': np.mean([ag.phi for ag in self.agents]),
            'total_phi': self.total_foam()
        }
