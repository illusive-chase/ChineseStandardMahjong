# -*- coding: utf-8 -*-

__all__ = ('merger',)

from inspect import getsource
import os

def merger(func):

    source = getsource(func).replace('@merger', '')
    arg_name = func.__code__.co_varnames[0]
    func_name = func.__code__.co_name

    def wrapped_func(args):
        if not args.merge:
            return func(args)

        generated = ''
        generated += 'from argparse import Namespace; ' + arg_name + ' = ' + repr(args) + '\n'
        generated += source + '\n'
        generated += 'if __name__ == "__main__":\n'
        generated += '    ' + func_name + '(' + arg_name + ')\n'
        with open('./__main__.py', 'w', encoding='utf-8') as f:
            f.write(generated)
        os.system('zip -r merge.zip ./__main__.py ./utils ./env ./learning')

    return wrapped_func