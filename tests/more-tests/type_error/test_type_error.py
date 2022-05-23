from pysd import external, Component

__pysd_version__ = "3.0.0"
_root = './'

component = Component()

external.ExtData('input.xlsx', 'Sheet1', '5', 'B6')
