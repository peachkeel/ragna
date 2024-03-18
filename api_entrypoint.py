import uvicorn
from ragna.deploy._api import app as api_app
from ragna.deploy._cli.config import parse_config

config = parse_config('./ragna.toml')

uvicorn.run(
        api_app(
            config=config, ignore_unavailable_components=False
        ),
        host=config.api.hostname,
        port=config.api.port,
    )
