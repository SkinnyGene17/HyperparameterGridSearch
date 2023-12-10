from tkinter import *
import group10_gridsearch_alg as gs

window=Tk()

def delete():
    entry1.delete(0,END)
    entry2.delete(0,END)
    entry3.delete(0,END)  ## USED TO CLEAR ROWS 
    result.delete(0,END)

def add():
    Optimizer=(entry1.get().split())
    Epochs=entry2.get().split()
    for i in range(len(Epochs)):
        Epochs[i]=int(Epochs[i])
    batchSize=entry3.get().split()
    for i in range(len(batchSize)):
        batchSize[i]=int(batchSize[i])
    d =  str(Optimizer) + " " + str(Epochs) + " "  + str(batchSize)  #METHOD ADDED TO TEST OUTPUT OF UI 
    result.insert(0,d) #TO GET USERINPUT USE entry1.get() FOR OPTIMIZER entry2.get()FOR EPOCHS 
                                                                                    #entry3.get() for BATCH SIZE
    gs.gridSearchScript(optimizers=Optimizer, epochs=Epochs, batches=batchSize)

mylabel=Label()
mylabel1=Label()

label1=Label(window,text="Optimizer: ", padx=20,pady=10)
label2=Label(window,text="Epochs: ", padx=20,pady=10)
label3=Label(window,text="Batch Size: ")
entry1=Entry(window,width=30,borderwidth=5)
entry2=Entry(window,width=30,borderwidth=5)
entry3=Entry(window,width=30,borderwidth=5)
add=Button(window,text="Add", padx=10,pady=5,command = add)
clear=Button(window,text="Clear",padx=10,pady=5,command=delete)
result =Entry(window,width=30,borderwidth=5)

label1.grid(row=0,column=0)
label2.grid(row=1,column=0)
label3.grid(row=2,column=0)
    
entry1.grid(row=0,column=1)
entry2.grid(row=1,column=1)
entry3.grid(row=2,column=1)
add.grid(row=3,column=0,columnspan=3)
clear.grid(row=3,column=1)
result.grid(row=4,column=0,columnspan=4)

window.mainloop()

