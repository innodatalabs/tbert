# The MIT License
# Copyright 2019 Innodata Labs and Mike Kroutikov
#
import json
from types import SimpleNamespace


def cmp_dict(d1, d2, ctx):
    k1 = set(d1.keys())
    k2 = set(d2.keys())
    if k1 != k2:
        ctx.error = 'dict keys mismatch'
        return

    for key in k1:
        cmp_x(d1[key], d2[key], ctx)
        if ctx.error is not None:
            ctx.path.append(key)
            return


def cmp_int(i1, i2, ctx):
    if i1 != i2:
        ctx.error = f'value mismatch {i1} vs {i2}'


def cmp_float(f1, f2, ctx):
    ctx.delta = max(abs(f1-f2), ctx.delta)
    if ctx.delta > ctx.tolerance:
        ctx.error = f'float value mismatch: {f1} vs {f2}'


def cmp_str(s1, s2, ctx):
    if s1 != s2:
        ctx.error = f'str value mismatch: {s1[:10]} vs {s2[:10]}'


def cmp_list(l1, l2, ctx):
    if len(l1) != len(l2):
        ctx.error = f'list length mismatch {len(l1)} vs {len(l2)}'
        return

    for index,(a,b) in enumerate(zip(l1, l2)):
        cmp_x(a, b, ctx)
        if ctx.error is not None:
            ctx.path.append(index)
            return


_DISPATCH = {
    int: cmp_int,
    float: cmp_float,
    str: cmp_str,
    dict: cmp_dict,
    list: cmp_list,
}

def cmp_x(a, b, ctx):
    if type(a) is not type(b):
        ctx.error = f'type mismatch: {type(a)} vs {type(b)}'
        return

    _DISPATCH[type(a)](a, b, ctx)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compares two JSON-L files')
    parser.add_argument('jsonl1', help='Path to first JSON-L file')
    parser.add_argument('jsonl2', help='Path to first JSON-L file')
    parser.add_argument('--tolerance', default=1.e-5, type=float, help='Float comparisom tolerance')

    args = parser.parse_args()

    with open(args.jsonl1, 'r', encoding='utf-8') as f1:
        with open(args.jsonl2, 'r', encoding='utf-8') as f2:

            f1 = iter(f1)
            f2 = iter(f2)

            ctx = SimpleNamespace(
                error=None,
                path=[],
                tolerance=args.tolerance,
                delta=0.
            )

            while True:
                l1 = next(f1, None)
                l2 = next(f2, None)

                if l1 is None:
                    if l2 is None:
                        break
                    print('Premature end of file 1')
                    break
                elif l2 is None:
                    print('Premature end of file 2')
                    break

                cmp_x(json.loads(l1), json.loads(l2), ctx)
                if ctx.error is not None:
                    path = '/'.join(str(x) for x in ctx.path)
                    print(ctx.error, 'at', path)
                    break

            print('Max float values delta:', ctx.delta)
            if not ctx.error:
                print('Structure is identical')
            parser.exit(-1 if ctx.error else 0)

