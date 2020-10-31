import argparse
import glob
import json
import os
import requests
from datetime import datetime

SCRIPT_URL = "https://script.google.com/macros/s/AKfycbzaG2qqiE9xSJwumi_gbPjDjYqe0_8tHXe_OApamvrnV0pk9uBR/exec"

REQUIRED_TAGS = {
    "split": None,
    "scatter": None,
    "overlay": None,
    "coefficient": None,
    "intercept": None,
    "features": None,
    "subset": None,
    "max": None,
    "x1_max": None,
    "x2_max": None
}

def get_jupyter_notebooks(extension="ipynb"):
    print('Reading files in directory...')
    return glob.glob("{0}/*.{1}".format(os.getcwd(), extension))

def read_notebook(filename):
    nb_submission = {}
    nb = json.load(open(filename))
    if 'cells' not in nb:
        return nb_submission
    cells = nb['cells']
    for cell in cells:
        if cell['metadata'] and 'tags' in cell['metadata']:
            source = ''.join(cell['source'])
            output = None
            if 'outputs' in cell:
                try:
                    output = next(map(
                        lambda output: ''.join(output['data']['text/plain']),
                        filter(
                            lambda output: output['output_type'] == 'execute_result',
                            cell['outputs']
                        )
                    ))
                except StopIteration:
                    pass
            for tag in cell['metadata']['tags']:
                nb_submission[tag] = source
                nb_submission['{0}_output'.format(tag)] = output
    print('Parsed', filename)
    return nb_submission

def get_sheetname(filename):
    return filename.split('/')[-1].split('.')[0]

def get_responses(filenames):
    if not filenames:
        filenames = get_jupyter_notebooks()
    return {get_sheetname(filename): read_notebook(filename) for filename in filenames}

def transform_sheet_data(sheet_data):
    """Do anything to change how the data will appear on the spreadsheet. Each key in the dictionary represents a different column"""
    try:
        transformed = {}
        for k, v in sheet_data.items():
            transformed[k] = v
        return transformed
    except Exception as e:
        return sheet_data

def is_submission_valid(sheet_data):
    for tag, placeholder in REQUIRED_TAGS.items():
        if not sheet_data.get(tag) or sheet_data[tag] == placeholder:
            print('\nCould not find submission for tag "{0}"'.format(tag))
            print('Maybe you forgot to save the notebook?')
            if 'output' in tag:
                print('Or forgot to run all cells?')
            return False
    return True

def submit(sub, verbose=False):
    user = sub['email']
    for sheet_name, sheet_data in sub['submission'].items():
        print('Posting answers for', sheet_name)
        sheet_data['email'] = user
        sheet_data['sheet'] = sheet_name
        sheet_data['timestamp'] = datetime.now()
        final_submission = transform_sheet_data(sheet_data)
        if not is_submission_valid(sheet_data):
            return False
        if verbose:
            print('Your submission:', final_submission)
        r = requests.post(
            SCRIPT_URL,
            data=final_submission
        )
        if not r.ok:
            return False
    return True

def create_and_submit(files=[], verbose=False):
    sub = get_responses(files)
    user = input('Enter your Berkeley email address: ')
    if submit({
        'email': user,
        'submission': sub
    }, verbose=verbose):
        print('\nSubmitted!')
    else:
        print('\nCould not submit, please try again later')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='*')

    args = parser.parse_args()
    files = args.files if 'files' in args else []

    submission = get_responses(files)

    email = 'araj@berkeley.edu'

    if submit({
        'email': email,
        'submission': submission
    }, verbose=True):
        print('Submitted!')
    else:
        print('Could not submit, please try again later')
