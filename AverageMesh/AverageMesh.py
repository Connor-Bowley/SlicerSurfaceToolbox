import itertools
import logging
import os
import pathlib
import typing
from typing import Annotated

import numpy as np

import qt
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import NodeModify, VTKObservationMixin

from slicer.parameterNodeWrapper import *
from MRMLCorePython import vtkMRMLModelNode


# https://stackoverflow.com/a/3844832
def allEqual(iterable):
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)

#
# AverageMesh
#

class AverageMesh(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Average Mesh"
        self.parent.categories = ["Surface Models"]
        self.parent.dependencies = []
        self.parent.contributors = ["Connor Bowley (Kitware, Inc.)"]
        self.parent.helpText = """
Given a set models in point correspondence, computes the average model, pointwise, optionally averaging scalars.
It is assumed that all models have the same number of points and the same cells.
"""
        self.parent.acknowledgementText = "Developed for SlicerSALT (salt.slicer.org)."

#
# AverageMeshParameterNode
#

@parameterNodeWrapper
class AverageMeshParameterNode:
    averageScalars: Annotated[bool, Default(True)]
    outputMesh: vtkMRMLModelNode
    activeTab: Annotated[int, Default(0)]
    inputDirectory: typing.Optional[pathlib.Path]
    #note: don't need to store the input meshes because they are queryable via this tag
    filterAttribute: Annotated[str, Default("AverageMeshWidgetIncludedTag")]

#
# AverageMeshWidget
#

class AverageMeshWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setFilterAttribute(self, attr):
        self._parameterNode.filterAttribute = attr
        self.ui.includedTree.sortFilterProxyModel().setIncludeNodeAttributeNamesFilter([self._parameterNode.filterAttribute])
        self.ui.excludedTree.sortFilterProxyModel().setExcludeNodeAttributeNamesFilter([self._parameterNode.filterAttribute])

    def filterAttribute(self):
        return self._parameterNode.filterAttribute

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/AverageMesh.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = AverageMeshLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.outputMeshComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputDirectoryButton.directoryChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.tabWidget.currentChanged.connect(self.updateParameterNodeFromGUI)

        # Buttons
        self.ui.averageScalarsCheckbox.clicked.connect(self.updateParameterNodeFromGUI)
        self.ui.includeButton.clicked.connect(self.onIncludeClicked)
        self.ui.excludeButton.clicked.connect(self.onExcludeClicked)
        self.ui.applyButton.clicked.connect(self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # setup the filter on the ui components
        self.setFilterAttribute(self.filterAttribute())

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        self.setParameterNode(AverageMeshParameterNode(self.logic.getParameterNode()))


    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        if inputParameterNode is None or isinstance(inputParameterNode, AverageMeshParameterNode):
            self._parameterNode = inputParameterNode
        else:
            self._parameterNode = AverageMeshParameterNode(inputParameterNode)

        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        try:
            # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
            self._updatingGUIFromParameterNode = True

            # if the inputDirectory is the default of None, set it to the last directory picked
            if self._parameterNode.inputDirectory is None:
                self._parameterNode.inputDirectory = pathlib.Path(qt.QSettings().value("AverageMesh/LastPickedDirectory", os.path.expanduser("~")))

            # Update the GUI
            self.ui.averageScalarsCheckbox.checked = self._parameterNode.averageScalars
            self.ui.outputMeshComboBox.setCurrentNode(self._parameterNode.outputMesh)

            self.ui.tabWidget.currentIndex = self._parameterNode.activeTab
            self.ui.inputDirectoryButton.directory = self._parameterNode.inputDirectory
        finally:
            # All the GUI updates are done
            self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Modify all properties in a single batch
        with NodeModify(self._parameterNode):
            self._parameterNode.averageScalars = self.ui.averageScalarsCheckbox.checked
            self._parameterNode.outputMesh = self.ui.outputMeshComboBox.currentNode()

            self._parameterNode.activeTab = self.ui.tabWidget.currentIndex
            self._parameterNode.inputDirectory = pathlib.Path(self.ui.inputDirectoryButton.directory)
            qt.QSettings().setValue("AverageMesh/LastPickedDirectory", str(self._parameterNode.inputDirectory) if self._parameterNode.inputDirectory is not None else None)

    @staticmethod
    def _getSelectedNodes(treeView):
        ids = vtk.vtkIdList()
        treeView.currentItems(ids)
        # Exclude the root item, since we don't want the scene object, just the nodes from the scene (if any).
        # currentItems returns the root item from currentItems if nothing is selected
        ids = [ids.GetId(i) for i in range(ids.GetNumberOfIds()) if ids.GetId(i) != treeView.rootItem()]
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        nodes = [shNode.GetItemDataNode(i) for i in ids]
        return nodes

    def onIncludeClicked(self):
        selectedNodes = self._getSelectedNodes(self.ui.excludedTree)
        for node in selectedNodes:
            node.SetAttribute(self.filterAttribute(), "true")
        self.refreshTreeViews()

    def onExcludeClicked(self):
        selectedNodes = self._getSelectedNodes(self.ui.includedTree)
        for node in selectedNodes:
            node.SetAttribute(self.filterAttribute(), None)
        self.refreshTreeViews()

    def refreshTreeViews(self):
        self.ui.excludedTree.sortFilterProxyModel().invalidateFilter()
        self.ui.includedTree.sortFilterProxyModel().invalidateFilter()

    def inputMeshesFromScene(self):
        return [x for x in slicer.util.getNodesByClass('vtkMRMLModelNode') if x.GetAttribute(self.filterAttribute()) is not None]

    def inputDirectory(self):
        return self._parameterNode.inputDirectory

    def loadInputMeshesFromFolder(self):
        inputDirectory = self.inputDirectory()
        if not os.path.isdir(inputDirectory):
            raise Exception(f'Input directory "{inputDirectory}" does not exist or is not a directory')

        try:
            result = []
            for filename in os.listdir(inputDirectory):
                filepath = os.path.join(inputDirectory, filename)
                if not os.path.isfile(filepath):
                    logging.warn(f'Skipping non-file item in inputDirectory: {filepath}')
                    continue
                result.append(slicer.util.loadModel(filepath))
                result[-1].CreateDefaultDisplayNodes()
                result[-1].GetDisplayNode().SetVisibility(False)
            return result
        except Exception as e:
            for model in result:
                slicer.mrmlScene.RemoveNode(model)
            raise

    def outputMesh(self):
        return self._parameterNode.outputMesh

    def averageScalars(self):
        return self._parameterNode.averageScalars

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            # Compute output
            if self.ui.tabWidget.currentWidget() == self.ui.inputFromSceneTab:
                models = self.inputMeshesFromScene()
                self.logic.process(models, self.outputMesh(), self.averageScalars())
                for model in models:
                    model.GetDisplayNode().SetVisibility(False)
            elif self.ui.tabWidget.currentWidget() == self.ui.inputFromFilesTab:
                try:
                    models = self.loadInputMeshesFromFolder()
                    self.logic.process(models, self.outputMesh(), self.averageScalars())
                finally:
                    for model in models:
                        slicer.mrmlScene.RemoveNode(model)
            else:
                raise Exception("Unknown tab: Indicates bug in AverageMeshWidget")
            if self.outputMesh().GetDisplayNode() is None:
                self.outputMesh().CreateDefaultDisplayNodes()
            self.outputMesh().GetDisplayNode().SetVisibility(True)


#
# AverageMeshLogic
#

class AverageMeshLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self._stdName = 'std'
        self._varianceName = 'variance'

    def standardDeviationArrayName(self):
        return self._stdName
    
    def varianceArrayName(self):
        return self._varianceName

    @staticmethod
    def _numpy_to_named_vtk(name, numpyArray):
        vtkArray = numpy_to_vtk(numpyArray)
        vtkArray.SetName(name)
        return vtkArray

    @staticmethod
    def _getArrayNames(pointOrCellData):
        return [pointOrCellData.GetArray(i).GetName()
                for i in range(pointOrCellData.GetNumberOfArrays())]

    @staticmethod
    def _hasDuplicateArrayNames(mesh):
        pointArrayNames = AverageMeshLogic._getArrayNames(mesh.GetMesh().GetPointData())
        cellArrayNames = AverageMeshLogic._getArrayNames(mesh.GetMesh().GetCellData())
        return len(pointArrayNames) != len(set(pointArrayNames)) or len(cellArrayNames) != len(set(cellArrayNames))

    @staticmethod
    def _divideCommon(pointOrCellDatas):
        dataNames = [AverageMeshLogic._getArrayNames(data) for data in pointOrCellDatas]
        if not dataNames:
            return ([], [])
        allUniquePointDataNames = list(set(name for names in dataNames for name in names))
        isCommonToAllMeshes = [all(uniqueName in dataNamesFromMesh for dataNamesFromMesh in dataNames)
                               for uniqueName in allUniquePointDataNames]
        commonNames = [allUniquePointDataNames[i] for i in range(len(allUniquePointDataNames)) if isCommonToAllMeshes[i]]
        uncommonNames = [allUniquePointDataNames[i] for i in range(len(allUniquePointDataNames)) if not isCommonToAllMeshes[i]]
        return (commonNames, uncommonNames)

    def _processScalars(self, scalarName, inputData, outputData):
        numpys = np.array([vtk_to_numpy(data.GetArray(scalarName))
                           for data in inputData])
        averages = numpys.sum(axis=0) / len(numpys)
        outputData.AddArray(self._numpy_to_named_vtk(scalarName, averages))
        outputData.AddArray(self._numpy_to_named_vtk(f"{scalarName}_{self._stdName}", numpys.std(axis=0)))
        outputData.AddArray(self._numpy_to_named_vtk(f"{scalarName}_{self._varianceName}", numpys.var(axis=0)))

    def process(self, inputMeshes, outputMesh, averageScalars=True):
        """
        Computes the average pointwise mesh for the inputs.
        inputMeshes -- The meshes to average, as vtkMRMLModelNodes.
        outputMesh -- The vtkMRMLModelNode to put the average mesh into.
        averageScalars -- Whether to average the scalars common to all input meshes.

        Assumes that all inputMeshes have the same cells.

        If averageScalars is True and not all inputMeshes have the same scalars, only scalars common to all meshes will
        be averaged.

        Will raise ValueError in the following cases:
        - There are no inputMeshes.
        - Any of the inputMeshes are None.
        - Not all inputMeshes have the same number of points.
        - The outputMesh is None (the outputMesh does not need to have a poly data set).
        - averageScalars is True and at least one input mesh has two point/cell data arrays with the same name.
        """
        import time
        startTime = time.time()

        if not inputMeshes:
            raise ValueError("No input meshes have been given")
        if any([x.GetMesh() is None for x in inputMeshes]):
            raise ValueError("A model without a mesh was given as input")
        if not allEqual([x.GetMesh().GetNumberOfPoints() for x in inputMeshes]):
            raise ValueError("Not all meshes have the same number of points")
        if not outputMesh:
            raise ValueError("No output mesh has been given")
        if averageScalars and any([self._hasDuplicateArrayNames(mesh) for mesh in inputMeshes]):
            raise ValueError("At least one input mesh has two point/cell data arrays with the same name")

        logging.info('AverageMeshLogic processing started')

        commonPointDataNames, uncommonPointDataNames = AverageMeshLogic._divideCommon(
            [mesh.GetMesh().GetPointData() for mesh in inputMeshes])
        commonCellDataNames, uncommonCellDataNames = AverageMeshLogic._divideCommon(
            [mesh.GetMesh().GetCellData() for mesh in inputMeshes])

        if (uncommonPointDataNames):
            logging.warn(f"Skipping point data scalars that don't appear in all input meshes: {uncommonPointDataNames}")
        if (uncommonCellDataNames):
            logging.warn(f"Skipping cell data scalars that don't appear in all input meshes: {uncommonCellDataNames}")

        polyData = vtk.vtkPolyData()
        polyData.DeepCopy(inputMeshes[0].GetMesh())

        # remove arrays. We will add the scalars back later
        while polyData.GetPointData().GetNumberOfArrays() > 0:
            polyData.GetPointData().RemoveArray(0)
        while polyData.GetCellData().GetNumberOfArrays() > 0:
            polyData.GetCellData().RemoveArray(0)

        if polyData.GetNumberOfPoints() > 0:
            numpys = np.array([
                vtk_to_numpy(mesh.GetPolyData().GetPoints().GetData()) if mesh.GetPolyData().GetPoints() else np.array([])
                for mesh in inputMeshes
            ])
            averages = numpys.sum(axis=0) / len(numpys)
            polyData.GetPoints().SetData(numpy_to_vtk(averages))
            polyData.GetPointData().AddArray(self._numpy_to_named_vtk(self._stdName, numpys.std(axis=0)))
            polyData.GetPointData().AddArray(self._numpy_to_named_vtk(self._varianceName, numpys.var(axis=0)))

        if averageScalars:
            for name in commonPointDataNames:
                self._processScalars(name, [mesh.GetMesh().GetPointData() for mesh in inputMeshes], polyData.GetPointData())
            for name in commonCellDataNames:
                self._processScalars(name, [mesh.GetMesh().GetCellData() for mesh in inputMeshes], polyData.GetCellData())
        outputMesh.SetAndObserveMesh(polyData)

        logging.info(f'AverageMeshLogic processing completed in {time.time()-startTime:.2f} seconds')
