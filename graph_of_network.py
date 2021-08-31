# -*- coding: utf-8 -*-
# @Time    : 2021/5/30
# @Author  : jiang yangfan

import numpy as np
from scipy.sparse import csc_matrix, dok_matrix
from scipy.sparse.linalg import splu, cg
from sksparse.cholmod import cholesky
# from scipy.sparse import save_npz, load_npz
# import line_profiler


class Edge(object):
    def __init__(self, sourceNodeNumber, sinkNodeNumber, branchValue, is_sourceNode=None):
        self.sourceNodeNumber = sourceNodeNumber
        self.sinkNodeNumber = sinkNodeNumber
        self.branchValue = branchValue
        self.is_sourceNode = is_sourceNode      # when Edge in GraphOfNetwork.edgeResNodeDict, need it to indicate if Node is sourceNode of Edge


class GraphOfNetwork(object):
    def __init__(self, method='LU', orderingMethod=None, nodes=0, edges=0, shorts=0, voltageSource=0, currentSource=0):
        self.name = ""
        self.nodes = nodes                      # Number of nodes in the network(include shorts)
        self.edges = edges                      # Number of branches in the network(include shorts)
        self.shorts = shorts                    # Number of shorts in the network(zero value voltage sources)
        self.voltageSource = voltageSource      # Number of Voltage Source(not include shorts)
        self.currentSource = currentSource      # Number of Current Source

        self.method = method                    # Solve method
        self.orderingMethod = orderingMethod    # Ordering method——"Cholesky": ["natural","amd","metis","nesdis","colamd","default","best"], "LU": ["NATURAL","COLAMD","MMD_ATA","MMD_AT_PLUS_A"]
        self.order = 0                          # Order of matrix
        self.sparseMatrix = None                # Sparse matrix for MNA system of equations
        self.currentVector = None               # Right hand side of Ax = b
        self.solVec = None                      # Solution Vector

        self.nodeDict = {}                      # dict{nodeNumber: [NodeName,……]}
        self.edgeResNodeDict = {}               # dict{nodeNumber:[Edge]}, including edges of type "R"
        self.currentList = []                   # list[Edge], including edge of type "I"
        self.voltNodeDict = {}                  # dict{voltNodeNumber: Voltage value}, for nodes of not short edge of type "V"

    def add_edge_to_graph(self, sourceNodeNumber, sinkNodeNumber, branchValue, edgeType):
        """
        Processing branch information by classification.
        """
        if edgeType == "I":
            self.currentSource += 1
            self.currentList.append(Edge(sourceNodeNumber, sinkNodeNumber, branchValue))
        elif edgeType == "R":
            branchValue = 1.0 / branchValue
            if sourceNodeNumber:
                edge = Edge(sourceNodeNumber, sinkNodeNumber, branchValue, is_sourceNode=True)
                self.edgeResNodeDict[sourceNodeNumber] = self.edgeResNodeDict.get(sourceNodeNumber, list()) + [edge]
            if sinkNodeNumber:
                edge = Edge(sourceNodeNumber, sinkNodeNumber, branchValue, is_sourceNode=False)
                self.edgeResNodeDict[sinkNodeNumber] = self.edgeResNodeDict.get(sinkNodeNumber, list()) + [edge]
        elif edgeType == 'V':
            self.voltageSource += 1
            if sinkNodeNumber == 0:
                self.voltNodeDict[sourceNodeNumber] = branchValue
            elif sourceNodeNumber == 0:
                self.voltNodeDict[sinkNodeNumber] = -branchValue
            else:
                raise NotImplementedError("V should be grounded!")

    def generate_nodeNumbers(self, sourceNodeName, sinkNodeName, nodesNumber, nodesList1, nodesList2, del_nodeNums, is_edgeShort=False):
        """
        Generates initial ordinal number for nodes
        """
        # 1: for short edge's nodes
        if is_edgeShort:
            self.shorts += 1
            # (1)one node has ordinal number, the other don't: choose the existed number for two nodes
            if sourceNodeName in nodesList1.keys() and sinkNodeName not in nodesList1.keys():
                self.nodes += 1

                sourceNodeNumber = nodesList1[sourceNodeName]
                sinkNodeNumber = sourceNodeNumber
                nodesList1[sinkNodeName] = sinkNodeNumber
                nodesList2[sinkNodeNumber] = nodesList2.get(sinkNodeNumber, list()) + [sinkNodeName]
            elif sourceNodeName not in nodesList1.keys() and sinkNodeName in nodesList1.keys():
                self.nodes += 1

                sinkNodeNumber = nodesList1[sinkNodeName]
                sourceNodeNumber = sinkNodeNumber
                nodesList1[sourceNodeName] = sourceNodeNumber
                nodesList2[sourceNodeNumber] = nodesList2.get(sourceNodeNumber, list()) + [sourceNodeName]
            # (2)both two nodes have ordinal number: delete the big number and choose the small number for two nodes
            elif sourceNodeName in nodesList1.keys() and sinkNodeName in nodesList1.keys():
                NodeNumber = min(nodesList1[sourceNodeName], nodesList1[sinkNodeName])
                del_nodeNum = max(nodesList1[sourceNodeName], nodesList1[sinkNodeName])
                del_nodeNums.append(del_nodeNum)
                # print("{} is del".format(del_nodeNum))
                if NodeNumber == nodesList1[sourceNodeName]:
                    nodesList2[NodeNumber] = nodesList2.get(NodeNumber, list()) + [sinkNodeName]
                    nodesList2[del_nodeNum].remove(sinkNodeName)
                    nodesList1[sinkNodeName] = NodeNumber
                else:
                    nodesList2[NodeNumber] = nodesList2.get(NodeNumber, list()) + [sourceNodeName]
                    nodesList2[del_nodeNum].remove(sourceNodeName)
                    nodesList1[sourceNodeName] = NodeNumber
            # (3)two nodes don't have ordinal number: generate one number for two nodes
            else:
                self.nodes += 2
                nodesNumber += 1
                nodesList1[sourceNodeName] = nodesNumber
                nodesList1[sinkNodeName] = nodesNumber
                nodesList2[nodesNumber] = nodesList2.get(nodesNumber, list()) + [sourceNodeName, sinkNodeName]
        # 2: for not short edge's node, generate two number for two nodes
        else:
            for NodeName in [sourceNodeName, sinkNodeName]:
                if NodeName not in nodesList1.keys():
                    self.nodes += 1
                    nodesNumber += 1
                    nodesList1[NodeName] = nodesNumber
                    nodesList2[nodesNumber] = nodesList2.get(nodesNumber, list()) + [NodeName]

        return nodesNumber, nodesList1, nodesList2, del_nodeNums

    def convert_network_into_graph(self, filename, filepath=""):
        """
        Preprocessing of Spice files and build the grid data structure.
        :return:
        """
        print("\n-----------Convert powerGridFile into graph-----------")
        print(filepath+filename)
        file = open(filepath+filename, 'r')
        self.name = filename.split(".")[0]

        nodesNumber = 0
        nodesList1 = {"0": 0}       # {nodeName:nodeNumber}
        nodesList2 = {0: ["0"]}     # {nodeNumber:[nodeName1,nodeName2,……]}
        del_nodeNums = []           # Some Nodenumbers do not have corresponding nodes due to short circuit
        temp = []                   # Temporarily stores information for non-short-circuited "V" edge: [[sourceNodeName, sinkNodeName, branchValue, edgeType],[……],……]
        for line in file.readlines():
            is_edgeShort = False
            if ".end" in line:
                break
            elif ".op" in line or "*" in line:
                continue
            else:
                self.edges += 1
                edgeName, sourceNodeName, sinkNodeName, branchValue = line.split()
                edgeType = edgeName[0].upper()
                branchValue = float(branchValue)
                if branchValue == 0.0 and edgeType == "V":
                    # print('short edge:', line)
                    is_edgeShort = True
                else:
                    temp.append([sourceNodeName, sinkNodeName, branchValue, edgeType])

                nodesNumber, nodesList1, nodesList2, del_nodeNums = self.generate_nodeNumbers(sourceNodeName, sinkNodeName,
                                                                                              nodesNumber, nodesList1, nodesList2, del_nodeNums, is_edgeShort)

        self.nodes += 1
        file.close()
        print("------------read powerGridFile over------------")

        # generate nodeDict from nodeList2,nodeList1
        if len(del_nodeNums):
            print("length of deleted nodeNumbers is not 0, but {}".format(len(del_nodeNums)))
            print("------------sort again for nodeNumber----------")
            nodesList3 = list(nodesList2.items())
            nodesList3.sort(key=lambda x: x[0], reverse=False)
            i = 0
            for num_nodeName in nodesList3:
                if num_nodeName[1]:
                    if num_nodeName[0] != i:
                        for nodeName in num_nodeName[1]:
                            nodesList1[nodeName] = i
                            self.nodeDict[i] = self.nodeDict.get(i, list()) + [nodeName]
                    else:
                        for nodeName in num_nodeName[1]:
                            self.nodeDict[i] = self.nodeDict.get(i, list()) + [nodeName]
                    i += 1
        else:
            for nodeName in nodesList1:
                self.nodeDict[nodesList1[nodeName]] = self.nodeDict.get(nodesList1[nodeName], list()) + [nodeName]

        # add edge to graph
        print("------------add edge to graph------------------")
        for edge in temp:
            sourceNodeName, sinkNodeName, branchValue, edgeType = edge
            sourceNodeNumber = nodesList1[sourceNodeName]
            sinkNodeNumber = nodesList1[sinkNodeName]
            if edgeType not in "RIV":
                raise TypeError("edge is not in proper format!")
            else:
                self.add_edge_to_graph(sourceNodeNumber, sinkNodeNumber, branchValue, edgeType)
                # print(sourceNodeName, ":", sourceNodeNumber, sinkNodeName, ":", sinkNodeNumber, branchValue, edgeType)

        print("Parse over!")
        print("Total number of Nodes(include shorts):", self.nodes)
        print("Total number of Edges(include shorts):", self.edges)
        print("Total number of short Edges = ", self.shorts)
        print("Total number of Current Source = ", self.currentSource)
        print("Total number of Voltage Source(not include shorts):", self.voltageSource)
        # print("nodesDict:\n", self.nodeDict)
        print("-------------------------------------------------------\n")

    def init_sparse_matrix(self):
        """
        Initialize the sparse matrix in DOK format.
        """
        print("-------------Initialize the sparse matrix--------------")
        order = self.nodes - self.shorts - 1
        if order > 0:
            self.order = order
            print("order:", order)
            self.sparseMatrix = dok_matrix((order, order))
            self.currentVector = np.zeros(order)
        else:
            raise ValueError("sparseMatrix's order is <= 0!")

    # @profile
    def fill_sparse_matrix(self):
        """
        Convert it into a linear system solving problem: Ax=b.
        Generate sparse matrix A in DOK format and then convert it to CSC format.
        """
        self.init_sparse_matrix()
        print("----------------fill the sparse matrix-----------------")

        # "V" edgeType
        print("for \"V\" edgeType")
        for nodeNumber, nodeVolt in self.voltNodeDict.items():
            self.sparseMatrix[nodeNumber - 1, nodeNumber - 1] = 1
            self.currentVector[nodeNumber - 1] = nodeVolt

        # 'I' edgeType
        print("for \"I\" edgeType")
        for edge in self.currentList:
            sourceNodeNo = edge.sourceNodeNumber
            sinkNodeNo = edge.sinkNodeNumber
            if sinkNodeNo == 0:
                self.currentVector[sourceNodeNo - 1] -= edge.branchValue
            elif sourceNodeNo == 0:
                self.currentVector[sinkNodeNo - 1] += edge.branchValue
            else:
                self.currentVector[sourceNodeNo - 1] -= edge.branchValue    # current out from sourceNode of I: -
                self.currentVector[sinkNodeNo - 1] += edge.branchValue      # current in to sinkNode of I: +

        # 'R' edgeType
        print("for \"R\" edgeType")
        for nodeNumber in self.edgeResNodeDict:
            if nodeNumber in self.voltNodeDict.keys():
                continue
            else:
                for edge in self.edgeResNodeDict[nodeNumber]:
                    if edge.is_sourceNode:
                        otherNodeNo = edge.sinkNodeNumber
                    else:
                        otherNodeNo = edge.sourceNodeNumber

                    self.sparseMatrix[nodeNumber - 1, nodeNumber - 1] += edge.branchValue
                    if otherNodeNo == 0:
                        continue
                    elif otherNodeNo in self.voltNodeDict.keys():
                        self.currentVector[nodeNumber - 1] += self.voltNodeDict[otherNodeNo] * edge.branchValue
                    else:
                        self.sparseMatrix[nodeNumber - 1, otherNodeNo - 1] -= edge.branchValue

        self.sparseMatrix = csc_matrix(self.sparseMatrix)
        # save_npz('sparse_matrix_{}.npz'.format(self.name), self.sparseMatrix)

        # ----------Display MNA matrix and current vector-----------
        # print("MNA sparseMatrix:")
        # print(self.sparseMatrix)
        # print("currentVector:")
        # print(self.currentVector)

    # @profile
    def node_voltage_solver(self):
        """
        Interface realization of solving module of linear equations
        """
        print("\n-----------------Node voltage solution-----------------")
        current_vector = self.currentVector
        matrix = self.sparseMatrix
        method = self.method
        ordering_method = self.orderingMethod

        if method == 'LU':
            print("--------------1: LU Decomposition--------------")
            if ordering_method is None:
                LU = splu(matrix)
            elif ordering_method == "MMD_AT_PLUS_A":
                LU = splu(matrix, permc_spec=ordering_method, diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
            else:
                LU = splu(matrix, permc_spec=ordering_method)
            self.solVec = LU.solve(current_vector)
        elif method == 'CG':
            print("------------2: Conjugate Gradient--------------")
            self.solVec, exitCode = cg(matrix, current_vector)
            if exitCode == 0:
                print('0 : successful')
            elif exitCode > 0:
                print('>0 : convergence to tolerance not achieved, number of iterations:{}'.format(exitCode))
            else:
                print('<0 : illegal input or breakdown')
        elif method == 'cholesky':
            print("----------3: Cholesky Decomposition------------")
            if ordering_method is None:
                factor = cholesky(matrix)
            else:
                factor = cholesky(matrix, ordering_method=ordering_method)
            self.solVec = factor(current_vector)
        else:
            raise NotImplementedError("no method \"{}\"".format(method))

    def print_solution(self, filepath=""):
        """
        Output node voltage values to the solution file
        """
        print("\n--------------------Print solution---------------------")
        print("---------------------Voltage-------------------")

        if self.orderingMethod is None:
            solution_file = open(filepath + self.name + '_' + self.method + ".solution", 'w')
        else:
            solution_file = open(filepath + self.name + '_' + self.method + '_' + self.orderingMethod + ".solution", 'w')

        # print(self.nodeDict)
        for nodeName in self.nodeDict[0]:
            solution_file.write("{}  {:e}\n".format(nodeName, 0))
        for i in range(len(self.nodeDict) - 1):
            for nodeName in self.nodeDict[i + 1]:
                solution_file.write("{}  {:e}\n".format(nodeName, self.solVec[i]))

        solution_file.close()
