def relu(z):
    if z < 0:
        return 0
    return z

print(relu(-5))
print(relu(-1))
print(relu(0))
print(relu(3))
print(relu(10))