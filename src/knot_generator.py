# -*- coding: utf-8 -*-
"""
Created on Mar 25 2023
author: yjianzhu

"""
import os
import sys
import numpy as np
import pandas as pd

# 对core文件夹下的文件进行处理，生成文件名列表
def get_file_list(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                file_list.append(os.path.join(root, file))
    return file_list

# 读取文件，返回数据，使用pandas
def open_file(file):
    # 读取文件中存的矩阵，分隔符为\t和空格，无表头
    data = pd.read_csv(file, sep='\s+', header=None)
    #print(data.values.shape)
    return data.values

# 定义生成更长纽结的函数,输入data为numpy数组，拓展长度至length
def knot_generator(data, length,type="open",mod="MC"):
    """生成开链or闭链纽结"""
    N = data.shape[0]

    # 生成开链纽结
    if type == "open":
        # 在链前后各添加(length-N)/2个坐标，链前从z轴负方向想data[0]靠近，链后从data[N-1]向z轴正方向靠近
        for i in range(int((length - N) / 2)):
            data = np.insert(data, 0, np.array([0, 0, -1])+data[0,:], axis=0)
            data = np.insert(data, N + 2*i + 1, np.array([0, 0, 1])+data[N+2*i,:], axis=0)
        return data
    # 生成闭链纽结
    elif type == "close":
        # 生成旋转矩阵，使得data[-1,:]与z轴正方向重合
        # 平移矩阵，使得data[0,:]与原点重合
        data-=data[0,:]

        vec=data[-1,:]
        xvec=vec/np.linalg.norm(vec)
        #找和vec与z轴正方向垂直的旋转轴
        dx      =  xvec[0]
        dy      =  xvec[1]
        dz      =  xvec[2]
        cosa    =  dz
        sina    =  np.sqrt(1-cosa*cosa)
        rxy     =  np.sqrt(dx*dx + dy*dy)
        ux      =  -dy/rxy
        uy      =   dx/rxy

        R=np.zeros((3,3))
        R[0,0]    =  cosa + ux * ux * (1-cosa)
        R[0,1]     =  ux * uy * (1-cosa) 
        R[0,2]     =  uy * sina
        R[1,0]     =  uy * ux *(1-cosa) 
        R[1,1]    =  cosa + uy*uy*(1-cosa)
        R[1,2]     =  - ux*sina
        R[2,0]     =  - uy*sina
        R[2,1]     =  ux*sina
        R[2,2]     =  cosa

        data=np.dot(data,R)
        
        if(mod=="MC"):
            # 在y轴上找一个点，这个点距离data[0,:]和data[-1,:]相等
            if((length-N)%2==0):
                x_0=np.sqrt(((length-N)/2)**2-(data[-1,2]/2-0.5)**2)
                y_0=0
                z_0=0.5+data[-1,2]/2

                # 在x0,y0,z0和data[-1,:]的距离输出
                #print(np.linalg.norm(data[-1,:]-np.array([x_0,y_0,z_0])))
                # 在x0,y0,z0和data[-1,:]之间间隔距离1取点
                newv=np.array([x_0,y_0,z_0])-data[-1,:]
                newv=newv/np.linalg.norm(newv)

                for i in range(int((length-N)/2)):
                    data=np.insert(data,N+i,data[-1,:]+newv,axis=0)
                # 再次在
                z_0=-0.5+data[N-1,2]/2
                newv=np.array([x_0,y_0,z_0])-data[0,:]
                newv=newv/np.linalg.norm(newv)
                for i in range(int((length-N)/2)):
                    data=np.insert(data,0,data[0,:]+newv,axis=0)
                return data
            else:
                half=(length-N)//2
                x_0=np.sqrt((half+1)**2-(data[-1,2]/2)**2)
                y_0=0
                z_0=data[-1,2]/2
                vec=np.array([x_0,y_0,z_0])-data[-1,:]
                vec=vec/np.linalg.norm(vec)
                for i in range(half):
                    data=np.insert(data,N+i,data[-1,:]+vec,axis=0)
                vec=np.array([x_0,y_0,z_0])-data[0,:]
                vec=vec/np.linalg.norm(vec)
                for i in range(half+1):
                    data=np.insert(data,0,data[0,:]+vec,axis=0)

                return data
        else:
            return 

# 定义写入xyz文件的函数
def write_xyz(data, filename):
    N = data.shape[0]
    with open(filename, 'w') as f:
        f.write(str(N) + '\n\n')
        for i in range(N):
            f.write('1' + '\t' + str(data[i, 0]) + '\t' + str(data[i, 1]) + '\t' + str(data[i, 2]) + '\n')

# 定义计算相邻两点间距离的函数
def distance(data, type="open"):
    if(type=="open"):
        N = data.shape[0]
        dis = np.zeros(N-1)
        for i in range(1,N):
            dis[i-1] = np.linalg.norm(data[i, :] - data[i - 1, :])
        return dis
    elif(type=="close"):
        N = data.shape[0]
        dis = np.zeros(N)
        for i in range(1,N):
            dis[i-1] = np.linalg.norm(data[i, :] - data[i - 1, :])
        dis[N-1]=np.linalg.norm(data[0,:]-data[N-1,:])
        return dis

# 定义保存为lammps input格式的函数
def write_lammps(data,filename,type="open",Lx=200,Ly=200,Lz=200):
    N=data.shape[0]
    with open(filename,'w') as f:
        f.write("#LAMMPS input file\n")
        f.write('{} atoms\n'.format(N))
        # 写入bond数目
        if (type=="open"):
            f.write('{} bonds\n'.format(N-1))
        elif(type=="close"):
            f.write('{} bonds\n'.format(N))
        # 写入angle数目
        if (type=="open"):
            f.write('{} angles\n'.format(N-2))
        elif(type=="close"):
            f.write('{} angles\n'.format(N))

        # 写入原子类型数目
        f.write('\n1 atom types\n')
        # 写入bond类型数目
        f.write('1 bond types\n')
        # 写入angle类型数目
        f.write('1 angle types\n')

        # 写入box的大小
        min_x=np.min(data[:,0])
        data[:,0] = data[:,0] - min_x
        min_y=np.min(data[:,1])
        data[:,1] = data[:,1] - min_y
        min_z=np.min(data[:,2])
        data[:,2] = data[:,2] - min_z
        f.write('\n0.0 {} xlo xhi\n'.format(max(Lx,np.max(data[:,0]))))
        f.write('0.0 {} ylo yhi\n'.format(max(Ly,np.max(data[:,1]))))
        f.write('0.0 {} zlo zhi\n'.format(max(Lz,np.max(data[:,2]))))

        # 写入质量
        f.write('\nMasses\n\n1 1.0\n')

        # 写入原子坐标
        f.write('\nAtoms\n\n')
        for i in range(N):
            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(i+1,1,1,data[i,0],data[i,1],data[i,2]))
        # 写入bond信息
        f.write('\nBonds\n\n')
        if (type=="open"):
            for i in range(N-1):
                f.write('{}\t{}\t{}\t{}\n'.format(i+1,1,i+1,i+2))
        elif(type=="close"):
            for i in range(N-1):
                f.write('{}\t{}\t{}\t{}\n'.format(i+1,1,i+1,i+2))
            f.write('{}\t{}\t{}\t{}\n'.format(N,1,N,1))
        # 写入angle信息
        f.write('\nAngles\n\n')
        if (type=="open"):
            for i in range(N-2):
                f.write('{}\t{}\t{}\t{}\t{}\n'.format(i+1,1,i+1,i+2,i+3))
        elif(type=="close"):
            for i in range(N-2):
                f.write('{}\t{}\t{}\t{}\t{}\n'.format(i+1,1,i+1,i+2,i+3))
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(N-1,1,N-1,N,1))
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(N,1,N,1,2))

# 定义读取xyz文件的函数
def read_xyz(filename):
    with open(filename,'r') as f:
        N=int(f.readline())
        f.readline()
        data=np.zeros((N,3))
        for i in range(N):
            line=f.readline().split()
            data[i,0]=float(line[1])
            data[i,1]=float(line[2])
            data[i,2]=float(line[3])
    return data


if __name__ == '__main__':
    knot_cores=get_file_list("core")
    types="close"
    Lknot=300

    for knot in knot_cores:
        data=open_file(knot)
        data=knot_generator(data,Lknot,types)
        # 从文件名中提取纽结类型
        knot_type=knot.split("_")[1]
        knot_type=knot_type.split(".")[0]
        write_lammps(data,"lammps/{}_L{}_{}.data".format(knot_type,data.shape[0],types),type=types)