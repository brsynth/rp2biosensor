from types import SimpleNamespace
from pathlib import Path
from filecmp import cmpfiles
from rp2biosensor.__main__ import run

RESULT_PATH = Path(__file__).resolve().parent/ 'data' / 'input' / 'rp2-results-lactate.csv'
SINK_PATH = Path(__file__).resolve().parent / 'data' / 'input' / 'sink-h2o2.csv'
OUTPUT_DIR_PATH = Path(__file__).resolve().parent / 'data' / 'output_dir'
OUTPUT_FILE_PATH = Path(__file__).resolve().parent / 'data' / 'output_file'

# Compare except with expected file except for SVGs
def filter_lines(line):
    """Filter unwanted lines
    
    This method is used when filtering lines with the filter method.
    """
    if line.lstrip().startswith('"svg": '):
        return False
    return True

def compare_files(ref_dir: Path, test_dir: Path, files_to_cmp: list):
    """Compare two set of files
    
    Files are compared 2 by 2, coming from 2 different directories
    """
    match = []
    mismatch = []
    for file in files_to_cmp:
        with open(ref_dir / file) as ref, \
             open(test_dir / file) as test:
            f1 = filter(filter_lines, ref)
            f2 = filter(filter_lines, test)
            if all(x == y for x,y in zip(f1, f2)):
                match.append(file)
            else:
                mismatch.append(file)
    return match, mismatch


def test_dir_output(tmpdir):
    temp_path = tmpdir / 'dir_case'  # tmpdir scope is session wised
    options = {
        "rp2_results": f"{RESULT_PATH}",
        "sink_file": f"{SINK_PATH}",
        "opath": f"{temp_path}",
        "otype": "dir",
        "ojson": None,
        "cache_dir": None
        }
    files_to_cmp = [
        'index.html',
        'network.json',
        'css/viewer.css',
        'js/viewer.js'
    ]
    args = SimpleNamespace(**options)
    run(args)

    match, mismatch = compare_files(
        ref_dir=OUTPUT_DIR_PATH,
        test_dir=temp_path,
        files_to_cmp=files_to_cmp
        )
    
    try:
        assert all(item in match for item in files_to_cmp)
    except AssertionError as e:
        print("Matched Files    : {}".format(match))
        print("Mismatched Files : {}".format(mismatch))
        raise e

def test_file_output(tmpdir):
    temp_path = tmpdir / 'file_case'  # tmpdir scope is session wised
    options = {
        "rp2_results": f"{RESULT_PATH}",
        "sink_file": f"{SINK_PATH}",
        "opath": f'{temp_path / "biosensor.html"}',
        "otype": "file",
        "ojson": None,
        "cache_dir": None
        }
    files_to_cmp = ['biosensor.html']
    args = SimpleNamespace(**options)
    run(args)

    match, mismatch = compare_files(ref_dir=OUTPUT_FILE_PATH, test_dir=temp_path, files_to_cmp=files_to_cmp)
    try:
        assert 'biosensor.html' in match
    except AssertionError as e:
        print("Matched Files    : {}".format(match))
        print("Mismatched Files : {}".format(mismatch))
        raise e