import os 
import torch 

# TODO: Use Hydra for better configuration management in later versions 

# Global configuration that serves for all the environments

GLOBAL_CONFIG = {
    "MODEL_PATH": "../model/model.pt",
    "USE_CUDE_IF_AVAILABLE": True,
    "ROUND_DIGIT": 3
}

# Environment specific configuration

ENV_CONFIG = {
    "development": {
        "DEBUG": True
    },

    "staging": {
        "DEBUG": True
    },

    "production": {
        "DEBUG": False,
        "ROUND_DIGIT": 3
    }
}


def get_config() -> dict: 
    ENV = os.environ['PYTHON_ENV'] if 'PYTHON_ENV' in os.environ else 'development'
    ENV = ENV or 'development'

    if ENV not in ENV_CONFIG:
        raise EnvironmentError(f'Config for envirnoment {ENV} not found')
    
    config = GLOBAL_CONFIG.copy() 
    config.update(ENV_CONFIG[ENV])

    config['ENV'] = ENV 
    config['DEVICE'] = 'cuda' if torch.cuda.is_available() and config['USE_CUDE_IF_AVAILABLE'] else 'cpu'

    return config 

CONFIG = get_config()

if __name__ == '__main__':
    import json 
    print(json.dumps(CONFIG, indent=4))