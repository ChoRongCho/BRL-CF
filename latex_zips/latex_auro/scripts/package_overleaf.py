"""Package active manuscript sources and dependencies for Overleaf.

Does not modify the packaged BRL_achitecture folder. Build artifacts, analysis
assets, backup manuscripts and review notes are excluded.
"""
from pathlib import Path
import argparse
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def collect():
    files = set()

    def add(path):
        if not path.is_file():
            raise FileNotFoundError(path)
        path.relative_to(ROOT)
        files.add(path)

    def tex(path):
        if path in files:
            return
        add(path)
        content = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text())
        for name in re.findall(r'\\(?:input|include)\s*\{([^}]+)\}', content):
            if name == 'Ustmry.fd':
                continue  # System font definition supplied by stmaryrd.
            target = ROOT/name
            if not target.suffix:
                target = target.with_suffix('.tex')
            tex(target)
        for name in re.findall(r'\\includegraphics\*?\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}', content):
            target = ROOT/name
            if not target.is_file():
                target = next((target.with_suffix(ext) for ext in ['.pdf','.png','.jpg','.jpeg']
                               if target.with_suffix(ext).is_file()), target)
            add(target)
        for names in re.findall(r'\\bibliography\s*\{([^}]+)\}', content):
            for name in names.split(','):
                add((ROOT/name.strip()).with_suffix('.bib'))
        for name in re.findall(r'\\documentclass\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}', content):
            local = ROOT/(name+'.cls')
            if local.exists(): add(local)
        for name in re.findall(r'\\(?:bibliographystyle|brlBibliographystyle)\s*\{([^}]+)\}', content):
            if '#' not in name: add(ROOT/(name+'.bst'))
    tex(ROOT/'main.tex')
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Destination ZIP file')
    args = parser.parse_args()
    files = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(ROOT)))
    print(f'Packaged {len(files)} files into {args.output}; main document: main.tex')

if __name__ == '__main__':
    main()
