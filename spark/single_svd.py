import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg.distributed import RowMatrix
import time
import matplotlib.pyplot as plt


conf = SparkConf()
conf.set("spark.master", "local[1]")  # 仅使用一个线程
conf.set("spark.driver.cores", "1")  # 限制驱动程序使用一个核心
conf.set("spark.executor.cores", "1")  # 限制执行器使用一个核心
conf.set("spark.executor.instances", "1")  # 限制执行器实例为一个
sc = SparkContext(conf=conf)

MIN_SIZE = 2
MAX_SIZE = 600
NUM_JOBS = 200
A = None

def job(row, col, rank):
    global A
    if A is None or len(A) != row * col:
        A = np.random.random((row, col)).tolist()
    rdd = sc.parallelize(A)
    mat = RowMatrix(rdd)
    start_time = time.time()
    mat.computeSVD(rank, computeU=True)
    return  time.time() - start_time


def test_size_time():
    time_cost_list = []
    mission_cost = 0
    for i in range(MIN_SIZE, MAX_SIZE + 1):
        cost = job(i, i, i)
        time_cost_list.append(cost)
        print(i, cost)
        mission_cost += cost

    print(f"Mission costs {mission_cost}s")

    plt.plot(range(MIN_SIZE, MAX_SIZE + 1), time_cost_list)
    plt.title(f"Single node costs {mission_cost}s")
    plt.xlabel("Matrix Size")
    plt.ylabel("Time/svd(s)")
    plt.savefig("Single_node_svd_size_time.png")


def test_time_numjobs():
    end_time_list = []
    cost_time_list = []
    mission_cost = 0
    for i in range(NUM_JOBS):
        cost = job(MAX_SIZE, MAX_SIZE, MAX_SIZE)
        mission_cost += cost
        end_time_list.append(mission_cost)
        cost_time_list.append(cost)
        print(i, cost)
    print(f"{NUM_JOBS} missions cost {mission_cost}s")
    throughout_list = 1.0 / np.array(cost_time_list)
    plt.plot(end_time_list, throughout_list)
    plt.title(f"Single node costs {mission_cost}s")
    plt.xlabel("TimeStamp(s)")
    plt.ylabel("Throughout(jobs)")
    plt.savefig("Single_node_svd_time_throughout.png")


if __name__ == "__main__":
    # ex1. matrixSize vs operationTime
    test_size_time()
    # ex2. time vs operationCount
    # test_time_numjobs()