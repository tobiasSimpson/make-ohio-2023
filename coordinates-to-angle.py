from math import *

lat = float(input("Lat: ")) * (2 * pi / 360)
long = float(input("Long: ")) * (2 * pi / 360)
time = float(input("Time (hours since midnight): "))
revolution = float(input("Rev: ")) * (2 * pi / 360)

long = (time - 12) * 2 * pi / 24 + revolution

EARTH_TILT = 23.4 * 2 * pi / 360

x = cos(long) * cos(lat)
y = sin(long) * cos(lat)
z = sin(lat)

(y, z) = (cos(EARTH_TILT) * y + -sin(EARTH_TILT) * z, sin(EARTH_TILT) * y + cos(EARTH_TILT) * z)

dot = x * cos(revolution) + y * sin(revolution)

if dot <= 0: print("dark side")
else:
    angle = (acos(dot) * 360 / (2 * pi))
    print(90 - angle)