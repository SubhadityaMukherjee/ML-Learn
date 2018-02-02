import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier,X,y):
    #min and max values in grid
    min_x,max_x = X[:,0].min()-1.0,X[:,0].max()+1.0
    min_y,max_y = y[:,1].min()-1.0,y[:,1].max()+1.0

    #step size in plotting mesh
    mesh_step_size = 0.01

    #mesh grid of X and Y values
    x_vals,y_vals=np.meshgrid(np.arange(min_x,max_x,mesh_step_size),np.arange(min_y,max_y,mesh_step_size))

    #To use classifier for the full meshgrid and then reshape it
    output = classifier.predict(np.c_[x_vals.ravel(),y_vals.ravel()])
    output=output.reshape(x_vals.shape)

    #Create the plot
    plt.figure()

    #Color scheme
    plt.pcolormesh(x_vals,y_vals,output,cmap=plt.cm.gray)

    #Overlay points on the plot
    plt.scatter(x[:,0],x[:,1],c=y,s=75,edgecolors="black",linewidth = 1,cmap=plt.cm.Paired)

    #Specify boundaries
    plt.xlim(x_vals.min(),x_vals.max())
    plt.xlim(y_vals.min(),y_vals.max())

    #Get ticks on X and Y axis
    plt.xticks((np.arange(int(X[:,0].min()-1),int(X[:,0].max()+1),1.0)))
    plt.yticks((np.arange(int(X[:,1].min()-1),int(X[:,1].max()+1),1.0)))

    #Show plot
    plt.show()
