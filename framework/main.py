from scripts import train, test
import yaml


def get_config(config_path):
    with open(config_path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


if __name__ == "__main__":
    train(config=get_config('./config.yaml'))
    # test(config=get_config('./config.yaml'))

