import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, Ridge, Lasso

np.random.seed(5)
trainDataRange = np.array([-3,6])

#True model: y=x^3-5x^2+3+9
a = 1
b = -5
c = 3
d = 9

#Generate random training data within the trainDataRange
#Parameter for showing plots and number of samples
def generateTrainData(numSamples=20,showPlot=True):
    #Generate X data
    trainX = np.linspace(trainDataRange[0],trainDataRange[1],numSamples)
    trainX = np.sort(trainX)
    trueY = transform_x(trainX)
    #Add noise with variance 10
    noisyY = trueY+np.random.randn(numSamples)*20
    #Plot the data
    if showPlot:
        plt.plot(trainX,noisyY,'bo')
        plt.plot(trainDataRange, transform_x(trainDataRange), 'r-', lw=2)
        plt.ylabel("Noisy Y")
        plt.xlabel("Input X")
        plt.title("Clean X and Noisy Y from Linear Relationship")
        plt.ylim(ymin=min(noisyY-20),ymax=max(noisyY)*2)
        plt.show()
        plt.clf()
    return trainX,noisyY

def transform_x(inputX):
    return a*np.power(inputX,3)+b*np.power(inputX,2)+c*np.power(inputX,1)+d

#Validate the data on a larger range
#Default is -2 to 9
#Training region is denoted by dotted lines
def validate(model,trainX,noisyY,numSamples=500,dataRange=[-4,7],showPlot=True,coeffs={},fig_size=(7,5)):
    dataRange=np.arange(dataRange[0],dataRange[1],0.1)
    transformed_dataRange = transform_x(dataRange)
    #Generate x values from the data range
    inputX = np.random.random_sample((numSamples,))*(dataRange[-1]-dataRange[0])+dataRange[0]
    inputX = np.sort(inputX)
    trueY = transform_x(dataRange)
    predY=predict(model,dataRange)
    #Plot graphs
    if showPlot:
        #A bit of math to determine where to draw the dotted lines
        plt.figure(figsize = fig_size)
        #plt.axes().clear()
        coordX1 = [trainDataRange[0]]*2
        coordX2 = [trainDataRange[1]]*2
        minY = min(min(transformed_dataRange),min(predY))
        maxY = max(max(transformed_dataRange),max(predY))
        line1, = plt.plot(coordX1, [minY,maxY], 'k-', lw=1,linestyle="--", label="Training Data Range")
        plt.plot(coordX2, [minY,maxY], 'k-', lw=1,linestyle="--")
        #TrainX
        train, = plt.plot(trainX,noisyY,'ko', lw=2, label="Training Data")
        #Prediction
        pred, = plt.plot(dataRange,predY,'b-', lw=2, label="Predicted Model")
        #True Data
        trueData, = plt.plot(dataRange, transform_x(dataRange),'r-', lw=1.5, label="True Data Distribution",alpha=0.7)

        plt.ylabel("Y")
        plt.xlabel("Input X")
        plt.ylim(ymin=min(noisyY-20),ymax=max(noisyY)*2)
        title = "Degree " + str (len(model)-1)+ " Model"
        for i,key in enumerate(coeffs.keys()):
            title += " "+str(key)+"="+str(coeffs[key])
            if i<len(coeffs.keys())-1:
                title +=","
        plt.title(title)
        # Create a legend for the first line.
        legend = plt.legend(handles=[line1,train,pred,trueData])
        plt.show()
    return error(trueY,predY)

def validateRegularization(model,predictFn,trainX,noisyY,l,numSamples=500,dataRange=[-4,7],showPlot=True,coeffs={},fig_size=(7,5)):
    dataRange=np.arange(dataRange[0],dataRange[1],0.1)
    transformed_dataRange = transform_x(dataRange)
    #Generate x values from the data range
    inputX = np.random.random_sample((numSamples,))*(dataRange[-1]-dataRange[0])+dataRange[0]
    inputX = np.sort(inputX)
    trueY = transform_x(dataRange)
    predY=predictFn(model,dataRange)
    #Plot graphs
    if showPlot:
        #A bit of math to determine where to draw the dotted lines
        plt.figure(figsize = fig_size)
        #plt.axes().clear()
        coordX1 = [trainDataRange[0]]*2
        coordX2 = [trainDataRange[1]]*2
        minY = min(min(transformed_dataRange),min(predY))
        maxY = max(max(transformed_dataRange),max(predY))
        line1, = plt.plot(coordX1, [minY,maxY], 'k-', lw=1,linestyle="--", label="Training Data Range")
        plt.plot(coordX2, [minY,maxY], 'k-', lw=1,linestyle="--")
        #TrainX
        train, = plt.plot(trainX,noisyY,'ko', lw=2, label="Training Data")
        #Prediction
        pred, = plt.plot(dataRange,predY,'b-', lw=2, label="Predicted Model")
        #True Data
        trueData, = plt.plot(dataRange, transform_x(dataRange),'r-', lw=1.5, label="True Data Distribution",alpha=0.7)

        plt.ylabel("Y")
        plt.xlabel("Input X")
        plt.ylim(ymin=min(noisyY-20),ymax=max(noisyY)*2)
        title = "Lambda = {0}".format(l)
        for i,key in enumerate(coeffs.keys()):
            title += " "+str(key)+"="+str(coeffs[key])
            if i<len(coeffs.keys())-1:
                title +=","
        plt.title(title)
        # Create a legend for the first line.
        legend = plt.legend(handles=[line1,train,pred,trueData])
        plt.show()
    return error(trueY,predY)

#Train the data
def model(trainX,trainY,degree=1):
    #Creates the vandermonde matrix https://en.wikipedia.org/wiki/Vandermonde_matrix
    powers=np.vander(trainX,degree+1)
    A=powers
    #Solves the normal equation
    model = np.linalg.solve(A.T@A,A.T@trainY)
    return model

def modelWithRidgeRegularization(trainX, trainY, regParam, degree=1):
    powers=np.vander(trainX,degree+1)
    A=powers
    #Solves the normal equation
    model = np.linalg.solve(A.T@A + regParam*np.eye(degree+1),A.T@trainY)
    return model

def modelWithLassoRegularization(trainX, trainY, regParam, degree=1):
    powers=np.vander(trainX,degree+1)
    A=powers
    #Solves the normal equation
    model = Lasso(alpha=regParam)
    model.fit(A,trainY)
    return model

    
#Predicts given x values based on a model
def predict(model,x):
    degree=len(model)-1
    powers=np.vander(x,degree+1)
    return powers@model

#Determines the error between true Y values and predicted
def error(trueY,predY):
    return np.linalg.norm((trueY-predY))/len(trueY)

#Generates graphs of different degree models
#Plots training error and test error
def overfittingDemo(degrees = [0,1,2,3,4,5,7,10,13]):
    trainX,trainY = generateTrainData(showPlot=False)
    trainError = []
    testError = []
    #Iterate over all model orders
    for deg in degrees:
        currModel = model(trainX,trainY,degree=deg)
        predTrainY = predict(currModel,trainX)
        currTrainErr = error(trainY,predTrainY)
        currTestErr = validate(currModel,trainX,trainY,showPlot=True)
        trainError.append(currTrainErr)
        testError.append(currTestErr)

def ridgeRegularizationDemo(lambdas, degree):
    trainX,trainY = generateTrainData(showPlot=False)
    trainError = []
    testError = []
    #Iterate over all model orders
    for l in lambdas:
        currModel = modelWithRidgeRegularization(trainX,trainY,l,degree=degree)
        predTrainY = predict(currModel,trainX)
        currTrainErr = error(trainY,predTrainY)
        currTestErr = validateRegularization(currModel,predict,trainX,trainY,l,showPlot=True)
        trainError.append(currTrainErr)
        testError.append(currTestErr)

def lassoRegularizationDemo(lambdas, degree):
    trainX,trainY = generateTrainData(showPlot=False)
    trainError = []
    testError = []
    #Iterate over all model orders
    for l in lambdas:
        currModel = modelWithLassoRegularization(trainX,trainY,l,degree=degree)
        predTrainY = currModel.predict(np.vander(trainX,degree+1))
        currTrainErr = error(trainY,predTrainY)
        currTestErr = validateRegularization(currModel,lambda model, x: model.predict(np.vander(x,degree+1)),trainX,trainY,l,showPlot=True)
        trainError.append(currTrainErr)
        testError.append(currTestErr)


    #Plot the errors
#     plt.figure(figsize=(10,4))
#     plt.subplot(1,2,1)
#     plt.plot(degrees,trainError)
#     plt.ylabel("Error")
#     plt.xlabel("Degree of Model")
#     plt.title("Training Error")


#     plt.subplot(1,2,2)
#     plt.plot(degrees,np.log(testError))
#     plt.title("Test Error")
#     plt.xlabel("Degree of Model")
#     plt.ylabel("Log Error")
#     plt.show()
#     Uncomment if you are curious about the actual error values
#     print("Training Errors:",trainError)
#     print("Test Errors:",testError)

def get_features(data, col_list, y_name):
    """
    Function to return a numpy matrix of pandas dataframe features, given k column names and a single y column
    Outputs X, a n X k dimensional numpy matrix, and Y, an n X 1 dimensional numpy matrix.
    This is not a smart function - although it does drop rows with NA values. It might break. 
    
    data(DataFrame): e.g. mpg, mpg_train
    col_list(list): list of columns to extract data from
    y_name(string): name of the column you to treat as the y column
    
    Ideally returns one np.array of shape (len(data), len(col_list)), and one of shape (len(data), len(col_list))
    """
    
    # keep track of numpy values
    feature_matrix = data[col_list + [y_name]].dropna().values
    return feature_matrix[:, :-1], feature_matrix[:, -1]    
    
def plot_multiple_linear_regression(data, x_1_name, x_2_name, y_name):
    """
    This function makes a 3D plot for multliple linear regression with two explanatory variables.
    Thanks to: https://stackoverflow.com/a/26434204
    
    data(DataFrame): e.g. mpg, mpg_train
    x_1_name(string): the name of the column representing the first explanatory variable
    x_2_name(string): the name of the column representing the second explanatory variable
    y_name(string): the name of the column representing the dependent/response variable
    
    returns None but outputs an interactive 3D plot
    """
    model = LinearRegression()
    X, Y = get_features(data, [x_1_name, x_2_name], y_name)
    fit = model.fit(X, Y)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    x_surf = np.arange(0, max(data[x_1_name]), max(data[x_1_name])/10)                # generate a mesh
    y_surf = np.arange(0, max(data[x_2_name]), max(data[x_2_name])/10)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)

    exog = pd.core.frame.DataFrame({x_1_name: x_surf.ravel(), x_2_name: y_surf.ravel()})
    out = fit.predict(exog)
    ax.plot_surface(x_surf, y_surf,
                    out.reshape(x_surf.shape),
                    rstride=1,
                    cstride=1,
                    color='None',
                    alpha = 0.4)

    ax.scatter(data[x_1_name], data[x_2_name], data[y_name],
               marker='o',
               alpha=1)

    ax.set_xlabel(x_1_name)
    ax.set_ylabel(x_2_name)
    ax.set_zlabel(y_name)
    plt.title("Linear Model with 2 Explanatory Variables: " + x_1_name + " and " + x_2_name + " vs. " + y_name, y=1.08, fontsize=16)

    plt.show();
    
