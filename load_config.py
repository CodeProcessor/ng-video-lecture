import yaml

from dotdict import dotdict


class ConfigClass:
    config = None

    def __init__(self):
        ConfigClass.config = self.load_config()

    def set_config(self, key, value):
        """Set configurations"""
        ConfigClass.config[key] = value

    def load_config(self, env='default'):
        # Load the YAML configuration
        with open('config.yaml', 'r') as f:
            yaml_config = yaml.safe_load(f)

        # Get the configuration for the specified environment
        if env not in yaml_config:
            raise ValueError(f"Environment '{env}' not found in the configuration.")
        if env != 'default':
            print(f"Using configuration for environment '{env}'.")

        config = yaml_config[env]
        return dotdict(config)


if __name__ == '__main__':
    config = ConfigClass()
    print(config.config.LEARNING_RATE)
    config.set_config('LEARNING_RATE', 0.01)
    print(config.config.LEARNING_RATE)
