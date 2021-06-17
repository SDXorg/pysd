from pysd import external

_root = './'

external.ExtData('input.xlsx', 'Sheet1', '5', 'B6',
                 None, {}, [], _root, '_ext_data')