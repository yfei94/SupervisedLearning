import random

f = open("data/sphere.csv", "w")

f.write("x,y,z,output\n")

for i in range(2000):
    # Generate x, y, z in the range [-1, 1]
    x =  random.random()*2 - 1
    y =  random.random()*2 - 1
    z =  random.random()*2 - 1

    output = 0
    if ((x**2 + y**2 + z**2 < 1)):
        output = 1

    f.write("%.3f,%.3f,%.3f,%d\n"%(x,y,z,output))

f.close()