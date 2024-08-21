from subprocess import Popen 
from time import time, sleep 

python2 = "python"
python3 = "python"

print("Starting Sat2Graph Server")

Popen("mkdir -p results", shell=True).wait()
fileServer = Popen("cd results; python -m SimpleHTTPServer 8000", shell=True)

mainServer = Popen("cd server; python main.py 8001 0", shell=True)


pWorker1 = Popen("cd worker; python localserverV2.py 8008", shell=True)
pWorker2 = Popen("cd worker; python localserverV2.py 8007", shell=True)
pWorker3 = Popen("cd worker; python localserverV1.py 8006", shell=True)
pWorker0 = Popen("cd worker; python localserverV1.py 8005", shell=True)
pWorker4 = Popen("cd worker; python localserverSeg.py 8009", shell=True)
pWorker5 = Popen("cd worker; python localserverSeg.py 8010", shell=True)
pWorker6 = Popen("cd worker; python localserverSeg.py 8011", shell=True)

fileServer.wait()