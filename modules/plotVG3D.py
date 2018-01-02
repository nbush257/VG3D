import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns



sns.set()
def plot3():
    f = plt.figure()
    return(f.add_subplot(111,projection='3d'))
def polar():
    f = plt.figure()
    return(f.add_subplot(111,projection='polar'))

