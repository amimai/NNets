import sys

def startProgress(title):
    global progress_x
    sys.stdout.write(title + ": [" + "-"*40 + "]" + chr(8)*41)
    sys.stdout.flush()
    progress_x = 0

def progress(x):
    global progress_x
    d = str(x)
    sys.stdout.write('  '+ str(d[0:min(5,len(d))]))
    sys.stdout.flush()
    progress_x = x

def endProgress():
    sys.stdout.write("#" + "]\n")
    sys.stdout.flush()