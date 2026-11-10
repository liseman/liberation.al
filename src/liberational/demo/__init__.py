"""Demo mode (#17): a small made-up household the dashboard can show without any real data.

`pack-demo` (demo/cli.py) makes a demo directory of its own, outside `data/`, seeds it with the dataset in
demo/dataset.py (three people, a pet, one place with a floor plan, and a day of journal entries), starts a private
Continuum broker on it, and runs the dashboard with every `paths.*` setting pointed there. Nothing reads or writes the
real data directory, and the README's screenshots are taken from it.
"""
