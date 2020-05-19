from pathlib import Path

import numpy as np
import vtk as vtklib
import vtk.util.numpy_support as vtknp


def vtu(filename, case, mu, lhs, lift=True, fields=dict()):
    tri = case.triangulation()
    nelems, _ = tri.shape

    _points = case.discretized_geometry(mu)
    npts, ndims = _points.shape
    if ndims < 3:
        _points = np.hstack([_points, np.zeros((npts, 3 - ndims))])
    points = vtklib.vtkPoints()
    points.SetData(vtknp.numpy_to_vtk(_points, deep=True))

    _cells = np.hstack([3 * np.ones((nelems, 1), dtype=int), tri])
    cells = vtklib.vtkCellArray()
    cells.SetCells(nelems, vtknp.numpy_to_vtkIdTypeArray(_cells.ravel(), deep=True))

    grid = vtklib.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.SetCells(vtklib.VTK_TRIANGLE, cells)

    for fieldname in case.bases.keys():
        _field = case.discretized(mu, lhs, fieldname, lift=lift)
        field = vtknp.numpy_to_vtk(_field, deep=True)
        field.SetName(fields.get(fieldname, fieldname))
        grid.GetPointData().AddArray(field)

    writer = vtklib.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()
