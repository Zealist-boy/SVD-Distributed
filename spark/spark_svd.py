import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg.distributed import RowMatrix
import time
import matplotlib.pyplot as plt


conf = SparkConf().setAppName("SVD").setMaster("spark://master:7077").setSparkHome("/root/spark")
sc = SparkContext.getOrCreate(conf=conf)
# sc = SparkContext(master="local", appName="SVD Example")

MIN_SIZE = 2
MAX_SIZE = 600
NUM_JOBS = 200
NUM_WORKERS=4
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
    plt.title(f"Multiple nodes cost {mission_cost}s")
    plt.xlabel("Matrix Size")
    plt.ylabel("Time/svd(s)")
    plt.savefig("Multi_nodes_svd_size_time.png")


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
    plt.title(f"Multiple nodes cost {mission_cost}s")
    plt.xlabel("TimeStamp(s)")
    plt.ylabel("Throughout(jobs)")
    plt.savefig("Multi_nodes_svd_time_throughout.png")


if __name__ == "__main__":
    # ex1. matrixSize vs operationTime
    #test_size_time()
    # ex2. time vs operationCount
    test_time_numjobs()
