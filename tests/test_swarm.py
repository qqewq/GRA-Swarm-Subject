import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import yaml
from core.swarm import Swarm

def test_collective_subject_formation():
    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['swarm']['num_agents'] = 5
    config['swarm']['connection_density'] = 0.8
    config['swarm']['seed_subject_index'] = 0
    config['swarm']['gra']['max_iterations'] = 500

    swarm = Swarm(config)
    swarm.run(verbose=False)
    stats = swarm.get_subjectivity_stats()
    assert stats['collective_subject'] == True, "Рой не стал коллективным субъектом"
    assert stats['num_individual_subjects'] >= 1, "Хотя бы лидер должен остаться субъектом"
    print("Тест пройден: коллективный субъект сформирован.")
