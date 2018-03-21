#Let's try communicating
import mmap
import contextlib
import numpy as np
import time
import sys
if sys.version_info > (3, 0):
    import pickle
else:
    import cPickle as pickle


FILESIZE = 1000

def consume():    
    
    with open("mmap_data.p","rb") as f:
        #every second write some new data
        while True:
            t0 = time.time()
            #let's pickle some numpy data...
            data = None

            with contextlib.closing(mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)) as m:
                data = pickle.load(m)

            print("read time = ",time.time()-t0)
            print(data)
            #sleep for 1 second
            time.sleep(1)


def produce():

    #set up file
    f = open("mmap_data.p","wb")
    f.write(FILESIZE*b'\0')
    f.close()
    
    t = 0

    with open("mmap_data.p","r+b") as f:
        
        while True:
            #every second write some new data
            print("Writer at t = ",t)
            #let's pickle some numpy data...
            data = np.arange(20)+t
        
            with contextlib.closing(mmap.mmap(f.fileno(),0,access=mmap.ACCESS_WRITE)) as m:
                m.seek(0) #rewind
                pickle.dump(data,m)
                m.flush()
            
            #sleep for 1 second
            time.sleep(1)
            t += 1
    


if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1] == "read":
        print("Starting reader")
        consume()
    else:
        print("Starting writer")
        produce()


