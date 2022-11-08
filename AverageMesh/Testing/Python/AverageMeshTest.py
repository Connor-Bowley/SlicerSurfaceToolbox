import numpy.testing as nptest

import vtk
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest

from AverageMesh import AverageMeshLogic

np2vtk = vtk.util.numpy_support.numpy_to_vtk
def vtk2np(v):
    if isinstance(v, vtk.vtkPoints):
        return vtk.util.numpy_support.vtk_to_numpy(v.GetData())
    if isinstance(v, vtk.vtkDataArray):
        return vtk.util.numpy_support.vtk_to_numpy(v)
    if isinstance(v, vtk.vtkCellArray):
        arr = []
        idList = vtk.vtkIdList()
        for i in range(v.GetNumberOfCells()):
            v.GetCell(i, idList)
            arr.append([idList.GetId(j) for j in range(idList.GetNumberOfIds())])
        return arr
    # hope for the best I guess
    return vtk.util.numpy_support.vtk_to_numpy(v)

def makePoints(listOfPoints):
    """Expecting a list of 3-item lists of doubles or a vtk.vtkPoints.
       If it is a vtkPoints object, will just return the input.
    """
    if isinstance(listOfPoints, vtk.vtkPoints):
        return listOfPoints

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(len(listOfPoints))
    for i, point in enumerate(listOfPoints):
        points.SetPoint(i, point)
    return points

def makeCells(listOfCells):
    if isinstance(listOfCells, vtk.vtkCellArray):
        return listOfCells
    cells = vtk.vtkCellArray()
    for cell in listOfCells:
        cells.InsertNextCell(len(cell))
        for pointId in cell:
            cells.InsertCellPoint(pointId)
    return cells

def dataArray(name, scalars):
    dataArray = np2vtk(scalars)
    dataArray.SetName(name)
    return dataArray

def model(name, points=None, polys=None, pointDataArrays=None, cellDataArrays=None):
    model = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', name)
    model.SetAndObserveMesh(vtk.vtkPolyData())
    if points is not None:
        model.GetMesh().SetPoints(makePoints(points))
    if polys is not None:
        model.GetMesh().SetPolys(makeCells(polys))
    if pointDataArrays is not None:
        for arr in pointDataArrays:
            model.GetMesh().GetPointData().AddArray(arr)
    if cellDataArrays is not None:
        for arr in cellDataArrays:
            model.GetMesh().GetCellData().AddArray(arr)
    return model

def scalarNames(pointOrCellData):
    return [pointOrCellData.GetArray(i).GetName() for i in range(pointOrCellData.GetNumberOfArrays())]

def hasAllExpectedScalars(expected, actual):
    """ Assumes there are no duplicates
    """
    pexpected = set(scalarNames(expected.GetMesh().GetPointData()))
    pactual   = set(scalarNames(actual.GetMesh().GetPointData()))
    cexpected = set(scalarNames(expected.GetMesh().GetCellData()))
    cactual   = set(scalarNames(actual.GetMesh().GetCellData()))
    return len(pexpected.intersection(pactual)) == len(pexpected) \
        and len(cexpected.intersection(cactual)) == len(cexpected)

def isMissingAllExpectedScalars(expected, actual):
    """ Assumes there are no duplicates
    """
    pexpected = set(scalarNames(expected.GetMesh().GetPointData()))
    pactual   = set(scalarNames(actual.GetMesh().GetPointData()))
    cexpected = set(scalarNames(expected.GetMesh().GetCellData()))
    cactual   = set(scalarNames(actual.GetMesh().GetCellData()))
    alwaysOnPointScalars = {"std", "variance"}
    return set(pexpected).intersection(set(pactual)) == alwaysOnPointScalars \
        and set(cexpected).intersection(set(cactual)) == set()

def assertPointsEqual(model1, model2):
    nptest.assert_almost_equal(vtk2np(model1.GetMesh().GetPoints()), vtk2np(model2.GetMesh().GetPoints()))

def assertCellsEqual(model1, model2):
    nptest.assert_equal(vtk2np(model1.GetMesh().GetPolys()), vtk2np(model2.GetMesh().GetPolys()))

def assertPointArrayEqual(arrayNameOrIndex, model1, model2):
    nptest.assert_almost_equal(
        vtk2np(model1.GetMesh().GetPointData().GetArray(arrayNameOrIndex)),
        vtk2np(model2.GetMesh().GetPointData().GetArray(arrayNameOrIndex)),
    )

def _impl_assertHasExpectedScalars(expected, actual):
    names = scalarNames(expected)
    for name in names:
         nptest.assert_almost_equal(
            vtk2np(expected.GetArray(name)),
            vtk2np(actual.GetArray(name)),
         )

def assertHasExpectedScalars(expected, actual):
    assert hasAllExpectedScalars(expected, actual)
    _impl_assertHasExpectedScalars(expected.GetMesh().GetPointData(), actual.GetMesh().GetPointData())
    _impl_assertHasExpectedScalars(expected.GetMesh().GetCellData(), actual.GetMesh().GetCellData())

#
# AverageMeshTest
#

class AverageMeshTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()
        self.output = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
        self.logic = AverageMeshLogic()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_exceptions()
        self.test_modelWithNoPoints_averageScalars()
        self.test_modelWithNoPoints_noAverageScalars()
        self.test_smallmesh_noscalars()
        self.test_oneInput_noscalars()
        self.test_oneInput_scalars()

    def test_exceptions(self):
        # disallow no inputs
        with self.assertRaises(ValueError):
          self.logic.process([], self.output, True)
        with self.assertRaises(ValueError):
          self.logic.process([], self.output, False)

        # disallow model without a mesh
        emptyModel = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
        with self.assertRaises(ValueError):
          self.logic.process([emptyModel], self.output, True)
        with self.assertRaises(ValueError):
          self.logic.process([emptyModel], self.output, False)
        slicer.mrmlScene.RemoveNode(emptyModel)

        # disallow different number of points
        inputs = [
            model('0', points=[[1,2,3]]),
            model('1', points=[[1,2,3], [1,2,3]]),
        ]
        with self.assertRaises(ValueError):
          self.logic.process(inputs, self.output, True)
        with self.assertRaises(ValueError):
          self.logic.process(inputs, self.output, False)

        # disallow no output
        with self.assertRaises(ValueError):
          self.logic.process(inputs, None, True)
        with self.assertRaises(ValueError):
          self.logic.process(inputs, None, False)

    def test_modelWithNoPoints_averageScalars(self):
        self.logic.process([model("input")], self.output, True)
        self.assertIsNotNone(self.output.GetMesh())
        self.assertEqual(0, self.output.GetMesh().GetNumberOfPoints())

    def test_modelWithNoPoints_noAverageScalars(self):
        self.logic.process([model("input")], self.output, False)
        self.assertIsNotNone(self.output.GetMesh())
        self.assertEqual(0, self.output.GetMesh().GetNumberOfPoints())

    def _impl_oneInput(self, averageScalars):
        points = [
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [0,1,2],
        ]
        polys = [[0,1,2], [0,1,3]]
        scalars = [dataArray("name", [5,6,7,8]), dataArray("name2", [2,6,9,0])]
        inputs = [
            model('0', points=points, polys=polys, pointDataArrays=scalars),
        ]
        # one mesh should basically pass through
        expected = model('0',
            points=points,
            polys=polys,
            pointDataArrays=[
                dataArray("std", [[0.0]*3]*4),
                dataArray("variance", [[0.0]*3]*4),
                dataArray("name", [5,6,7,8]),
                dataArray("name_std", [0.0]*4),
                dataArray("name_variance", [0.0]*4),
                dataArray("name2", [2,6,9,0]),
                dataArray("name2_std", [0.0]*4),
                dataArray("name2_variance", [0.0]*4),
            ])

        self.logic.process(inputs, self.output, averageScalars)

        # common checks
        assertPointsEqual(expected, self.output)
        assertCellsEqual(expected, self.output)
        assertPointArrayEqual("std", expected, self.output)
        assertPointArrayEqual("variance", expected, self.output)
        return expected

    def test_oneInput_noscalars(self):
        expected = self._impl_oneInput(averageScalars=False)
        self.assertTrue(isMissingAllExpectedScalars(expected, self.output))

    def test_oneInput_scalars(self):
        expected = self._impl_oneInput(averageScalars=True)
        assertHasExpectedScalars(expected, self.output)

    def _impl_smallmesh(self, averageScalars):
        polys = [[0,1,2]]
        pointScalars0 = [dataArray("name", [5,6,7]), dataArray("name2", [5,6,7])]
        cellScalars0 = [dataArray("cname", [7])]
        pointScalars1 = [dataArray("name", [8,9,10])]
        cellScalars1 = [dataArray("cname", [9])]
        inputs = [
            model('0', points=[[1,2,3]], polys=polys, pointDataArrays=pointScalars0, cellDataArrays=cellScalars0),
            model('1', points=[[2,3,4]], polys=polys, pointDataArrays=pointScalars1, cellDataArrays=cellScalars1),
        ]
        expected = model('expected',
            points=[[1.5,2.5,3.5]],
            polys=polys,
            pointDataArrays=[
                dataArray("std", [[0.5, 0.5, 0.5]]),
                dataArray("variance", [[0.25, 0.25, 0.25]]),
                # note: name2 is excluded because it only shows up in one mesh
                dataArray("name", [6.5, 7.5, 8.5]),
                dataArray("name_std", [1.5, 1.5, 1.5]),
                dataArray("name_variance", [2.25, 2.25, 2.25]),
            ],
            cellDataArrays=[
                dataArray("cname", [8]),
            ],
        )
        self.logic.process(inputs, self.output, averageScalars)

        # common checks
        assertPointsEqual(expected, self.output)
        assertCellsEqual(expected, self.output)
        assertPointArrayEqual("std", expected, self.output)
        assertPointArrayEqual("variance", expected, self.output)
        return expected


    def test_smallmesh_noscalars(self):
        expected = self._impl_smallmesh(averageScalars=False)
        self.assertTrue(isMissingAllExpectedScalars(expected, self.output))

    def test_smallmesh_scalars(self):
        expected = self._impl_smallmesh(averageScalars=True)
        assertHasExpectedScalars(expected, self.output)
