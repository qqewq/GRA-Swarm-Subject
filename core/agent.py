import numpy as np
from typing import Optional

class Agent:
    """
    ИИ-агент с моделью мира M, слоем субъектности S и активным познанием A.
    Состояние: Ψ = (M, S, A).
    """
    def __init__(self, dim: int, initial_foam: float = 0.8, subjectivity_threshold: float = 0.05):
        self.dim = dim
        # Модель мира – нормированный вектор
        self.M = np.random.randn(dim)
        self.M /= np.linalg.norm(self.M)
        # Слой субъектности – скалярное "качество" субъектности, зависящее от пены
        self.S = 0.0  # 0 = отсутствует, 1 = полный субъект
        # Активное познание – матрица внимания/обработки
        self.A = np.eye(dim) * 0.1
        # Функционал внутренней пены
        self.phi = initial_foam
        self.threshold = subjectivity_threshold

    def update_subjectivity(self):
        """Обновляет слой субъектности по текущему уровню пены."""
        self.S = max(0.0, 1.0 - self.phi / self.threshold) if self.phi < self.threshold else 0.0

    def is_subject(self) -> bool:
        return self.phi < self.threshold

    def set_as_initial_subject(self):
        """Превращает агента в 'чистого' субъекта (Φ≈0)."""
        self.phi = 0.0
        self.S = 1.0
        # Идеально непротиворечивая модель мира (единичный базисный вектор)
        self.M = np.zeros(self.dim)
        self.M[0] = 1.0
        self.A = np.eye(self.dim) * 0.1

    def __repr__(self):
        return f"Agent(phi={self.phi:.4f}, S={self.S:.2f})"
