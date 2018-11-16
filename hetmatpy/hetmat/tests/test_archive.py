import filecmp
import pathlib
import zipfile

import hetmatpy.hetmat
import hetmatpy.hetmat.archive
import hetmatpy.matrix
from hetmatpy.tests.hetnets import get_graph


def test_disease_gene_example_hetmat_archiving(tmpdir):
    """
    Test archiving the hetmat corresponding to the hetnet in Figure 2C at https://doi.org/crz8.
    """
    tmpdir = pathlib.Path(tmpdir)
    graph = get_graph('disease-gene-example')
    hetmat_0_dir = tmpdir.joinpath('disease-gene-example-0.hetmat')
    hetmat = hetmatpy.hetmat.hetmat_from_graph(graph, hetmat_0_dir)

    # Test creating archive
    archive_path = hetmatpy.hetmat.archive.create_hetmat_archive(hetmat)
    with zipfile.ZipFile(archive_path) as zip_file:
        name_list = zip_file.namelist()
    expected = [
        'edges/DlT.sparse.npz',
        'edges/GaD.sparse.npz',
        'edges/GeT.sparse.npz',
        'edges/GiG.sparse.npz',
        'metagraph.json',
        'nodes/Disease.tsv',
        'nodes/Gene.tsv',
        'nodes/Tissue.tsv',
    ]
    assert name_list == expected

    # Test round-tripped hetmat has same files
    hetmat_1_dir = tmpdir.joinpath('disease-gene-example-1.hetmat')
    hetmatpy.hetmat.archive.load_archive(archive_path, hetmat_1_dir)
    match, mismatch, errors = filecmp.cmpfiles(hetmat_0_dir, hetmat_1_dir, common=expected, shallow=False)
    assert match == expected
    assert not mismatch
    assert not errors
