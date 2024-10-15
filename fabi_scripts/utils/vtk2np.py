import argparse
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa


def vtk_to_np(fname):
    reader = vtk.vtkPolyDataReader()

    reader.SetFileName(fname)
    reader.Update()
    polydata = reader.GetOutput()
    numpy_array_of_points = dsa.WrapDataObject(polydata).Points

    return numpy_array_of_points


def add_parser_args(parsel):
    parsel.add_argument('vtkfile', help='input .vtk (path)')
    parsel.add_argument('npfile', help='output .npy (path)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_parser_args(parser)
    args = parser.parse_args()

    # load and convert
    np_arr = vtk_to_np(args.vtkfile)

    # save out
    np.save(args.npfile, np_arr)
