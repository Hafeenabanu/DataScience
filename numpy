Numpy

1. import numpy as np

num1=([[2,4,5],[2,6,1]])

num2=([[3,4,7,],[2,6,1]])

num3=np.multiply(num1,num2)

print(num3)

2. import numpy as np

num1=([[2,4,5],[2,6,1]])

num2=([[3,4,7,],[2,6,1]])

print("a>b")

print(np.greater(num1,num2))

print(np.greater_equal(num1,num2))

print(np.less(num1,num2))

print(np.less_equal(num1,num2))

print(np.equal(num1,num2))

3. import numpy as np

a=np.arange(30,70,2)

print(a)

4. import numpy as np

a=np.identity(3,dtype=int)

print(a)

5. import numpy as np

a=np.arange(21)

a[(a>15)&(a<19)]*=-1

print(a)

6. import numpy as np

a=np.diag([1,2,3,4,5])

print(a)

7. import numpy as np

a=np.array([[1,2,3],[1,2,3]])

print(np.sum(a))

print(np.sum(a,axis=0))

print(np.sum(a,axis=1))

8. import numpy as np

a=np.arange(12).reshape(4,3)

print(a)

header='col1,col2,col3'

np.savetxt("temp.txt",a,header=header)

res=np.loadtxt("temp.txt")

print(res)

10.import numpy as np

a=np.array([1,2,3])

b=np.array([1,4,3])

print(np.equal(a,b))

11. import numpy as np

a=np.arange(16,dtype=int).reshape(-1,4)

print(a)

a[[0,-1],:]=a[[-1,0],:]

print(a)

matplotlib

1. import numpy as np

import matplotlib.pyplot as plt

x=np.array([1,2,6,8])

y=np.array([3,8,1,10])

plt.plot(x,y,marker='o',mec='g',mfc='g')

plt.show

2. import numpy as np

import matplotlib.pyplot as plt

x=np.array([12,14,16,18,20,22])

y=np.array([100,200,300,400,500,600])

plt.plot(x,y)

plt.title("Title")

plt.xlabel("Temperature in degree Celsius ")

plt.ylabel("sales")

plt.show()

4. import numpy as np

import matplotlib.pyplot as plt

x=np.array([1,2,3])

y=np.array([1,4,5])

plt.plot(x,y,label="line1")

x1=np.array([5,7,9])

y2=np.array([3,6,2])

plt.plot(x1,y2,label="line 2")

plt.legend()

plt.show()

5. import matplotlib.pyplot as plt

fig=plt.figure()

plt.subplot(2,2,1)

plt.xticks(())

plt.yticks(())

plt.subplot(2,3,4)

plt.xticks(())

plt.yticks(())

plt.show()

6. import matplotlib.pyplot as plt

import numpy as np

a=np.array(["java","python","php"])

b=np.array([22.8,8.6,11.6])

plt.xlabel("Programming lang")

plt.ylabel("poplularity")

plt.bar(a,b)

plt.show()

plt.barh(a,b,color="red")

plt.show()

plt.bar(a,b,color=["red","blue","green"])

plt.show()

la=["java","python","php"]

plt.pie(b,labels=la)

plt.show()

7. import numpy as np

import matplotlib.pyplot as plt

a=np.array([22, 30, 35, 35, 26])

b=np.array([25, 32, 30, 35, 29])

w=0.40

plt.bar(a-0.2,b,w,color="green")

plt.bar(a+0.2,b,w,color="red")

plt.legend(["male","female"])

plt.show()

Pandas

1.import pandas as pd

l=[1,2,4,5,7]

print(l)

r=pd.Series(l)

print(r)

2. import pandas as pd

p=pd.date_range(start='2022-5-1',end='2022-5-12')

for v in p:

print(v)

3. import pandas as pd

p={

'Name':["madhu","gopi","kuttappu"],

'roll':[1,2,3]

}

r=pd.DataFrame(p)

print(r)

4. import pandas as pd

p=[["madhu","gopi","kuttappu"],

[1,2,3]]

r=pd.DataFrame(p)

print(r)

5. import pandas as pd

df = pd.read_csv("CardioGoodFitness.csv")

print(df.head())

6. import pandas as pd

d=pd.DataFrame(

{

'x':[1,2,3],

'y':[12,3,4]

}

)

print(d)

print(d.set_index('x'))

7.
