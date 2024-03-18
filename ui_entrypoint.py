from ragna.deploy._cli.core import ui
from ragna.deploy._cli.config import parse_config
ui(config=parse_config('./ragna.toml'))