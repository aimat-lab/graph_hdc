import os

import rich_click as click
from rich_click import RichGroup
from pycomex.cli import ExperimentCLI

from graph_hdc.utils import PATH
from graph_hdc.utils import get_version
from graph_hdc.utils import CsvString

click.rich_click.USE_MARKDOWN = True


@click.group(invoke_without_command=True)
@click.option('-v', '--version', is_flag=True, required=False, help='Print the version of the package')
def cli(version):
    
    if version:
        print(get_version())
        return 0
    
    return 0
    

experiment_cli = ExperimentCLI(
    name='exp',
    experiments_path=os.path.join(PATH, 'experiments'),
    version=get_version()
)

cli.add_command(experiment_cli)


if __name__ == '__main__':
    cli()
