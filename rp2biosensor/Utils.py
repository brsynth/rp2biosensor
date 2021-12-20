from pathlib import Path
from distutils import dir_util
from tempfile import TemporaryDirectory


def write(args, template_dir, json_str: str):
    if args.otype == 'dir':
        # Prepare output dir
        outdir_path = Path(args.opath).resolve()
        outdir_path.mkdir(parents=True, exist_ok=True)
        # Copy template dir
        dir_util.copy_tree(str(template_dir), str(outdir_path))
        # Append network
        with open(outdir_path / 'network.json', 'w', newline='\n') as ofh:
            ofh.write(f'network = {json_str}')
    elif args.otype == 'file':
        # Prepare output dir
        outdir_path = Path(args.opath).resolve().parent
        outdir_path.mkdir(parents=True, exist_ok=True)
        outfile_path = Path(args.opath).resolve()
        # Copy template dir into a temporary folder
        with TemporaryDirectory() as temp_dir:
            tempdir_path = Path(temp_dir)
            dir_util.copy_tree(str(template_dir), str(tempdir_path))
            with open(tempdir_path / 'network.json', 'w', newline='\n') as ofh:
                ofh.write(f'network = {json_str}')
            html_str = all_in_one_file(tempdir_path)
        with open(outfile_path, 'wb') as ofh:
            ofh.write(html_str.replace(b'\r\n', b'\n'))
    else:
        raise NotImplementedError(f'Unexpected otype: {args.otype}')


def all_in_one_file(ifolder: Path) -> str:
    """Merge all needed file into a single HTML

    :param ifolder: folder containing the files to be merged
    :return html_str: string, the HTML
    """
    # find and open the index file
    html_str = open(ifolder / 'index.html', 'rb').read()
    # open and read JS files and replace them in the HTML
    js_to_replace = [
        'js/jquery-3.6.0.min.js',
        'js/cytoscape-3.19.0.min.js',
        'js/jquery-ui-1.12.1.min.js',
        'js/viewer.js'
    ]
    for js_file in js_to_replace:
        js_bytes = open(ifolder / js_file, 'rb').read()
        ori = b'src="' + js_file.encode() + b'">'
        rep = b'>' + js_bytes
        html_str = html_str.replace(ori, rep)
    # open and read style.css and replace it in the HTML
    css_to_replace = ['css/viewer.css']
    for css_file in css_to_replace:
        css_str = open(ifolder / css_file, 'rb').read()
        ori = b'<link href="' + css_file.encode() + b'" rel="stylesheet" type="text/css"/>'
        rep = b'<style type="text/css">' + css_str + b'</style>'
        html_str = html_str.replace(ori, rep)
    ### replace the network
    net_str = open(ifolder / 'network.json', 'rb').read()
    ori = b'src="' + 'network.json'.encode() + b'">'
    rep = b'>' + net_str
    html_str = html_str.replace(ori, rep)
    return html_str
