import datetime
import os

print("Hello from Spark Code! " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("Files in current directory:")
for item in os.listdir():
    print(item)