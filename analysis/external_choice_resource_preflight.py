"""Inspect licensed external resources without running downloaded research code."""
from pathlib import Path
import argparse, ast, hashlib, json
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    receipts = json.loads((a.input_dir/'download_receipt.json').read_text())
    for row in receipts:
        raw = (a.input_dir/row['record']/row['file']).read_bytes()
        assert hashlib.md5(raw).hexdigest() == row['publisher_md5'].split(':')[1]
        assert hashlib.sha256(raw).hexdigest() == row['sha256']
        assert row['license'] == 'cc-by-4.0'
    metadata = json.loads((a.input_dir/'23143985_metadata.json').read_text())
    available = {z['key'] for z in metadata['files']}
    author_code = ast.parse((a.input_dir/'23143985/analise_extra.py').read_text())
    imports = [z.name for n in ast.walk(author_code) if isinstance(n, ast.Import) for z in n.names]
    required = []
    for n in ast.walk(author_code):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr.startswith('read_'):
            required.extend(z.value for arg in n.args for z in ast.walk(arg)
                            if isinstance(z, ast.Constant) and isinstance(z.value, str))
    schemas = {}
    for csv in sorted((a.input_dir/'23143985').glob('*.csv')):
        # This file has a pandas two-row column header. We inspect the schema,
        # rather than treating header rows as additional numerical observations.
        frame = pd.read_csv(csv, header=[0,1], index_col=0) if csv.name=='extra_A_brier_por_cobertura.csv' else pd.read_csv(csv)
        schemas[csv.name] = {'parsed_rows':len(frame), 'columns':[str(c) for c in frame.columns],
                            'content_level':'aggregate context/replication results, not individual answer vectors'}
    path = a.input_dir/'15420092/COSCBB.xlsx'
    survey = pd.read_excel(path, 'survey'); choices = pd.read_excel(path, 'choices')
    assert choices[['unique_id', 'choice_card', 'option']].notna().all().all()
    assert not choices.duplicated(['unique_id','choice_card','option']).any()
    groups = choices.groupby(['unique_id', 'choice_card'])
    assert groups.size().eq(3).all() and groups.choice.sum().eq(1).all()
    assert set(choices.option.unique()) == {1,2,3}
    assert set(choices.choice.unique()) == {0,1}
    assert set(choices.unique_id).issubset(set(survey.unique_id))
    assert groups.option.nunique().eq(3).all()
    for column in ['ingr','nutr','label','access','prox','cost']:
        assert choices[column].notna().all()
    assert choices.loc[choices.option.eq(3), 'cost'].eq(0).all()
    tasks = groups.size().reset_index().groupby('unique_id').size()
    assert int(tasks.sum())*3 == len(choices)
    selected = choices.loc[choices.choice.eq(1)]
    by_person = selected.groupby('unique_id').option.nunique()
    constant = by_person.eq(1)
    constant_complete = int((constant & tasks.eq(6)).sum())
    result = {
        'date': '2026-11-09', 'file_checks': receipts,
        'source_23143985': {
            'doi': '10.5281/zenodo.23143985', 'version': metadata['metadata'].get('version'),
            'inspection': 'Current seven-file release and downloaded supplemental analysis source; no execution of downloaded code',
            'released_files': sorted(available),
            'released_CSV_schemas': schemas,
            'missing_local_module': 'estudo2_pipeline.py' if 'estudo2_pipeline' in imports and 'estudo2_pipeline.py' not in available else None,
            'referenced_inputs_absent_from_current_release': sorted(set(required)-available),
            'raw_AI_panel_vectors_in_current_file_list': False,
            'use': 'Related exploratory estimand/prediction work, not an executed independent replication; no claim about older releases or authors private files'},
        'source_15420092': {
            'doi': '10.5281/zenodo.15420092', 'file': 'COSCBB.xlsx',
            'survey_records': len(survey), 'choice_records': int(choices.unique_id.nunique()),
            'alternative_rows': len(choices), 'observed_tasks': int(len(groups)),
            'task_counts_per_record': {str(k): int(v) for k,v in tasks.value_counts().items()},
            'task_key': ['unique_id','choice_card'],
            'index_is_row_counter_not_task_number': True,
            'alternatives_per_task': 3, 'one_selected_per_task': True,
            'outside_option_code': 3, 'complete_constant_choice_records': constant_complete,
            'partial_constant_choice_records': int(constant.sum())-constant_complete,
            'attribute_columns': ['ingr','nutr','label','access','prox','cost'],
            'license': 'cc-by-4.0',
            'use': 'Available human DCE resource; codebook and basic structure verified. No preference model or external validation has been run.'},
        'raw_external_records_republished': False, 'downloaded_code_executed': False,
        'new_AI_generation': False, 'submission_or_novelty_proof': False}
    (a.out/'external_resource_preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'files_verified':len(receipts),'observed_external_tasks':len(groups),
        'task_counts_per_person':result['source_15420092']['task_counts_per_record'],
        'complete_constant_choice_records':constant_complete,
        'external_models_fitted':False,'downloaded_code_executed':False}))


if __name__ == '__main__':
    main()
