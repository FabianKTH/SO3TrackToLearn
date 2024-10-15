import vtk
from vtk.numpy_interface import dataset_adapter as dsa

def get_ico_points(subdiv=4):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(f'/fabi_project/sphere/ico{subdiv}.vtk')
    # reader.SetFileName(f'/mnt/3_tract_rl/sphere/ico{subdiv}.vtk')
    reader.Update()
    polydata = reader.GetOutput()
    numpy_array_of_points = dsa.WrapDataObject(polydata).Points

    return numpy_array_of_points