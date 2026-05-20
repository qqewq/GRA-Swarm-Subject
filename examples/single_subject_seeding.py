import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import yaml
from core.swarm import Swarm
import matplotlib.pyplot as plt

def main():
    with open('../configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Убедимся, что рой из 10 агентов, один субъект-лидер
    config['swarm']['num_agents'] = 10
    config['swarm']['seed_subject_index'] = 0

    swarm = Swarm(config)
    print("Начальная общая Φ:", swarm.total_foam())
    print("Агент-субъект (индекс 0):", swarm.agents[0])

    history = swarm.run(verbose=True)

    # Визуализация
    plt.figure(figsize=(10, 5))
    plt.plot(history, linewidth=2)
    plt.yscale('log')
    plt.xlabel('Итерация')
    plt.ylabel('Общая пена Φ')
    plt.title('Формирование коллективного субъекта в рое')
    plt.grid(True, alpha=0.3)
    plt.savefig('swarm_subject_formation.png', dpi=150)
    plt.show()

    stats = swarm.get_subjectivity_stats()
    print("\nИтоговая статистика:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Покажем, сколько агентов стали индивидуальными субъектами
    individual_status = [(i, ag.is_subject(), ag.phi) for i, ag in enumerate(swarm.agents)]
    print("\nСтатус агентов:")
    for idx, is_subj, phi in individual_status:
        print(f"  Агент {idx}: субъект={is_subj}, Φ={phi:.6f}")

if __name__ == "__main__":
    main()
