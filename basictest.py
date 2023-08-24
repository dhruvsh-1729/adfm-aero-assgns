import math as m

def transformation(h,arr):
  return [h*arr[1], h*arr[2], -1/2*h*arr[0]*arr[2]]

def runge_kutta(h,arr):
  k1 = transformation(h,arr)
  g_1 = [arr[i] + k1[i] for i in range(len(arr))]
  k2 = transformation(2*h,g_1)
  g_2 = [arr[i] + k2[i] for i in range(len(arr))]
  k3 = transformation(2*h,g_2)
  g_3 = [arr[i] + k3[i] for i in range(len(arr))]
  k4 = transformation(h,g_3)
  
  increment = [k1[i] + k2[i] + k3[i] + k4[i] for i in range(len(k1))]
  
  next = [arr[i] + increment[i] for i in range(len(arr))]
  return next

l=0.330491125
h=0.4
n=6
h=0.1
i=0
temp1=[0,0,l]
temp2 = [0,0,h]

threshold = 0.0001
err=1

def find(h,arr,n,thresh):
  i=0
  temp=arr
  while i<=n:
    next = runge_kutta(h,temp)
    if(m.fabs(next[2]-1) <= thresh):
      print(next)
      break
    temp = next
    i+=h
  return next

i=0
while i<=n:
  next = runge_kutta(h,temp1)
  if(m.fabs(next[2]-1)<=threshold):
    print(next,i)
    break
  temp1=next
  i+=h

print(next)