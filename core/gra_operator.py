import numpy as np

class GRAOperator:
    """
    Оператор обнуления N, уменьшающий функционал пены Φ(Ψ).
    Φ(Ψ) = ||M - A @ M||^2 + λ * (1 - S)^2, где λ – коэффициент важности субъектности.
    """
    def __init__(self, lambda_subj: float = 0.1, lr: float = 0.01):
        self.lambda_subj = lambda_subj
        self.lr = lr

    def compute_phi(self, agent) -> float:
        """Вычисляет внутреннюю пену агента."""
        M, A, S = agent.M, agent.A, agent.S
        # Рассогласование между моделью мира и её преобразованием активным познанием
        discrepancy = M - A @ M
        phi_raw = np.linalg.norm(discrepancy) ** 2
        # Штраф за отсутствие субъектности (чем ближе S к 1, тем меньше вклад)
        phi_subj = self.lambda_subj * (1 - S) ** 2
        return phi_raw + phi_subj

    def step(self, agent) -> None:
        """Одна итерация градиентного спуска по Φ для агента."""
        # Градиенты (упрощённые, через автоматическое дифференцирование можно точнее)
        # dΦ/dM = 2 (I - A)^T (M - A M)
        grad_M = 2 * (np.eye(agent.dim) - agent.A).T @ (agent.M - agent.A @ agent.M)
        # dΦ/dA = -2 (M - A M) @ M^T
        grad_A = -2 * np.outer(agent.M - agent.A @ agent.M, agent.M)

        agent.M -= self.lr * grad_M
        agent.A -= self.lr * grad_A

        # Нормируем M, чтобы избежать тривиального нуля
        norm_M = np.linalg.norm(agent.M)
        if norm_M > 1e-8:
            agent.M /= norm_M

        # Пересчитываем пену и субъектность
        agent.phi = self.compute_phi(agent)
        agent.update_subjectivity()
