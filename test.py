a = [1,2,3,4,5,6,7,8]
b = {'a':1, 'b':2, 'c':3}

print(a, type(a))
print(b, type(b))

num = [0]

for i in a:
    num[0] += 1
    if num[0] > a[0] and num[0] < a[-1]:
        print(666)
if 'a' in b:
    print(888)