# -*- coding: utf-8 -*-
import scipy.stats as stats
from graph_of_network import GraphOfNetwork
# import matplotlib.pyplot as plt
# import line_profiler


def correlation_calc_ibmpg(solution, calc_solution):
    print("\n-----Correlation coefficient calc for ibmpg------")
    file1 = open(solution, 'r')
    file2 = open(calc_solution, 'r')
    list1 = []
    list2 = []
    for line in file1.readlines():
        if line.strip("\n"):
            value = line.strip("\n").split("  ")[1]
            list1.append(eval(value))
    for line in file2.readlines():
        if line.strip("\n"):
            value = line.strip("\n").split("  ")[1]
            list2.append(eval(value))

    file1.close()
    file2.close()

    list1.sort()
    list2.sort()
    co = stats.pearsonr(list1, list2)[0]
    print("The correlation coefficient between {} and {} is:{}".format(solution, calc_solution, co))

    return co


# @profile
def test():
    print("------------------------START-------------------------")
    filepath = "./example/data/"
    output_path = "./example/output/"
    filename = "ibmpg1.spice"
    methods = ["LU", "CG", "cholesky"]
    method = methods[0]
    graph = GraphOfNetwork(method)
    graph.convert_network_into_graph(filename, filepath)
    graph.fill_sparse_matrix()
    # plt.spy(graph.sparseMatrix)
    # plt.show()
    # savemat('{}.mat'.format(graph.name), {'A': graph.sparseMatrix})
    graph.node_voltage_solver()
    graph.print_solution(output_path)
    if "ibmpg" in filename:
        solution = filepath + filename[:6]+".solution"
        calc_solution = output_path + filename[:6]+"_"+method+".solution"
        correlation_calc_ibmpg(solution, calc_solution)
    print("-------------------------END--------------------------")


if __name__ == '__main__':
    test()
