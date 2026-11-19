"""Authenticate reconstructed prompt examples against hashed author-cache keys."""
import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from source_data import dependencies, example_keys, prompt_pool, sha


def harmonize(name):
    value = str(name).strip()
    return 'label' if value == 'labl' else value


def workbook(core, context, excluded, config):
    book = pd.ExcelFile(BytesIO(core.read(context + '.xlsx')))
    choices = pd.read_excel(book, 'choices' if 'choices' in book.sheet_names else 'choice')
    choices.columns = list(map(harmonize, choices.columns))
    price = next(c for c in choices if c.lower().startswith(tuple(config['PRECO_CANDIDATOS'])))
    attributes = [c for c in choices if c not in excluded and c != price and pd.api.types.is_numeric_dtype(choices[c])]
    columns = attributes + [price]
    meta = pd.read_excel(book, 'metadata', header=None)
    product = meta.iloc[0, 0].split('choice experiment on')[-1].strip()
    names, values = {}, {}
    for _, detail in meta.iterrows():
        name, description, _, encoding = detail.iloc[:4]
        if not isinstance(name, str):
            continue
        name = harmonize(name)
        if isinstance(description, str) and description.startswith('Attribute'):
            names[name] = description.split(':', 1)[1].strip()
        for code in encoding.split('; ') if isinstance(encoding, str) else ():
            if ': ' in code:
                key, text = code.split(': ', 1)
                try:
                    values[name, float(key)] = text.strip().replace('No lebel', 'No label')
                except ValueError:
                    continue
    values.update({(name, 3.): '3 or more' for name in config['HH']})

    def display(column, value):
        try:
            return values.get((column, float(value)), str(value))
        except (TypeError, ValueError):
            return str(value)

    menus = {}
    for key, rows in choices.groupby(['unique_id', 'choice_card']):
        if len(rows) != 3:
            continue
        menu = []
        for _, row in rows.sort_values('option').iterrows():
            if row['option'] == 3:
                menu.append('I would not buy either of these products.')
            else:
                menu.append('; '.join(f"{names.get(c, c).split('.')[0].strip()}: {display(c, row[c])}" for c in columns))
        menus[key] = menu
    descriptions = '\n'.join('- ' + names.get(c, c) for c in columns)
    system = (f'You predict how a specific consumer answered one task of a discrete choice survey on {product}. '
              'Each task shows two product options and an option to buy neither. The attributes were described to '
              f'respondents as follows:\n{descriptions}\n\nReply with one letter only: A, B or C.')
    return system, menus, display


def verify(core, predictions, config, meta, cache, origin):
    letter = config['LETRAS']
    permutations = config['PERMS']
    cyclic = config['PERMS_CICLICAS']
    covered = set()
    max_error = 0.

    def menu(key, order):
        return '\n'.join(f'Option {letter[j]}: {menus[key][alt]}' for j, alt in enumerate(order))

    for context, context_rows in predictions.groupby('contexto', sort=True):
        _, tasks, _, _ = origin.read_table(core, context)
        system, menus, display = workbook(core, context, origin.EXCLUDE, config)
        assert set(menus) == set(tasks)
        for replica, split in context_rows.groupby('rep', sort=True):
            replica = int(replica)
            people, h, v, donors = prompt_pool(core, context, tasks, replica)
            profile = {}
            for person, row in people.iterrows():
                profile[person] = '; '.join(f"{config['NOME_COV'][c]}: " +
                                            ('not reported' if row[c] == -99 else display(c, row[c])) for c in people.columns)
            for condition, rows in split.groupby('cond', sort=True):
                examples = example_keys(people, donors[condition], np.r_[h, v], tasks, replica)
                prefixes = {}
                for person, keys in examples.items():
                    recorded = []
                    for key in keys:
                        order = permutations[(sum(map(ord, str(key))) + replica) % len(permutations)]
                        label = letter[order.index(tasks[key][1])]
                        recorded.append(f'Respondent profile: {profile[key[0]]}\n{menu(key, order)}\nAnswer: {label}')
                    prefixes[person] = ('Choices made by other respondents with similar profiles:\n\n' +
                                        '\n\n'.join(recorded) + f'\n\nTarget respondent profile: {profile[person]}')
                for row in rows.itertuples(index=False):
                    total = np.zeros(3)
                    for order in cyclic:
                        question = (menu((row.pid, row.card), order) +
                                    '\nWhich option did the target respondent choose? Answer with A, B or C only.')
                        parts = ('local', meta['modelo'], 'lp', system, prefixes[row.pid], question, '')
                        key = sha256('\x1f'.join(parts).encode()).hexdigest()
                        assert key in cache, (context, replica, condition)
                        covered.add(key)
                        total[order] += cache[key]
                    assert row.n_validas == len(cyclic) and row.n_invalidas == 0 and row.prob_nativa == 1
                    max_error = max(max_error, float(np.max(abs(total / len(cyclic) - [row.p1, row.p2, row.p3]))))
    assert max_error < 1e-12
    return covered, max_error


def main(args):
    plan, backend = dependencies(args.dependencies)
    origin = backend.external
    for name, expected in origin.HASHES.items():
        assert sha(args.inputs / name) == expected
    with ZipFile(args.inputs / origin.CORE) as core, ZipFile(args.inputs / origin.OUTPUT) as output:
        pipeline = next(n for n in core.namelist() if n.endswith('estudo2_pipeline.py'))
        assert sha256(core.read(pipeline)).hexdigest() == plan['sources']['required_original_pipeline_sha256']

        def payload(name):
            return output.read(next(n for n in output.namelist() if n.endswith('/' + name)))

        frozen = json.loads(payload('congelamento.json'))
        meta = json.loads(payload('previsoes_llm_meta.json'))
        assert meta['modo'] == 'logprobs' and meta['k'] == 3
        prediction_data = payload('previsoes_llm.csv')
        assert sha256(prediction_data).hexdigest() == frozen['hash_previsoes']
        records = map(json.loads, payload('cache_llm.jsonl').splitlines())
        cache = {r['h']: r['probs_pos'] for r in records if r.get('probs_pos')}
        predictions = pd.read_csv(BytesIO(prediction_data))
        covered, error = verify(core, predictions, frozen['config'], meta, cache, origin)
    assert len(predictions) == 7200
    receipt = dict(status='passed', reconstructed_requests=3*len(predictions),
                   distinct_matched_keys=len(covered), archived_distinct_keys=len(cache),
                   verified_probabilities=len(predictions), maximum_absolute_probability_error=error)
    (args.out / 'prompt_cache_summary.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'dependencies', 'out'):
        cli.add_argument('--' + name, type=Path, required=True)
    main(cli.parse_args())
