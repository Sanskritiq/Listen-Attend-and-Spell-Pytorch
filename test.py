import os
import pickle


folder='checkpoint/las_imp_libri_listener/data/'
db = {}  
for filename in os.listdir(folder):  
    if os.path.isfile(folder+filename):
        myfile = open(folder+filename,"rb")
        db[os.path.splitext(filename)[0]]= pickle.load(myfile)
        myfile.close()
        print(filename)

folder='checkpoint/las_imp_libri_listener/'    
for filename in os.listdir(folder):    
    if os.path.isfile(folder+filename):
        myfile = open(folder+filename,"rb")
        db[os.path.splitext(filename)[0]]= pickle.load(myfile)
        myfile.close()
        print(filename)

print(db)
myfile = open("merge/merge.pkl","wb")
pickle.dump(db, myfile)
myfile.close()
