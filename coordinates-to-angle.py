from math import *

# (lat, long) = map(float, input().split(", "))

lat = float(input("Lat: ")) * (2 * pi / 360)
long = float(input("Long: ")) * (2 * pi / 360)
time = float(input("Time (hours since midnight): "));

long += time * 2 * pi / 24

EARTH_TILT = 23.4 * 2 * pi / 360

x = cos(long) * cos(lat)
y = sin(long) * cos(lat)
z = sin(lat)

print(x, y, z)

angle = (acos(x) * 360 / (2 * pi))
print(angle)