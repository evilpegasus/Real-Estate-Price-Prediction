import argparse
import glob
import json
import os
import requests
from datetime import datetime

SCRIPT_URL = "https://script.google.com/macros/s/AKfycbxm8535zuydzAV54kB4Z6SLEvq9oBg5GWM6EittgUOY7T6nI_g/exec"

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
            nb_submission[cell['metadata']['tags'][0]] = ''.join(cell['source'])
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
            if k == 'name':
                transformed['name'] = v.replace('**Name**:', '').strip()
            elif k == 'major':
                transformed['major'] = v.replace('**Major**:', '').strip()
            elif k == 'fun-fact':
                transformed['fun-fact'] = v.replace('**Fun Fact**:', '').strip()
            else:
                transformed[k] = v
        return transformed
    except Exception as e:
        return sheet_data

def submit(sub):
    user = sub['email']
    for sheet_name, sheet_data in sub['submission'].items():
        print('Posting answers for', sheet_name)
        sheet_data['email'] = user
        sheet_data['sheet'] = sheet_name
        sheet_data['timestamp'] = datetime.now()
        r = requests.post(
            SCRIPT_URL,
            data=transform_sheet_data(sheet_data)
        )
        if not r.ok:
            return False
    return True

def create_and_submit(notebooks=[]):
    sub = get_responses(notebooks)
    user = input('Enter your Berkeley email address: ')
    if submit({
        'email': user,
        'submission': sub
    }):
        print('Submitted!')
    else:
        print('Could not submit, please try again later')

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
    }):
        print('Submitted!')
    else:
        print('Could not submit, please try again later')
