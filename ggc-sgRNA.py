
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import spearmanr
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

sequencelength=0
limiter=False
limiterstart=0
limiterend=0
Xtrain_raw=""
Xtest_raw=""
fileout = open("output_golden_gate_models_test.txt","w+")

# Conversion of input nucleotide sequences (ACGT-format) into binary values -- one-hot encoding
def convert_base(arrayinput):
    arrayoutput = []
    for sequence in arrayinput:
        onehotencoding = []
        for i in range(len(sequence)):
            if sequence[i].upper() == "A":
                onehotencoding.append([1,0,0,0])
            elif sequence[i].upper() == "C":
                onehotencoding.append([0,1,0,0])
            elif sequence[i].upper() == "G":
                onehotencoding.append([0,0,1,0])
            elif sequence[i].upper() == "T":
                onehotencoding.append([0,0,0,1])
            elif sequence[i].upper() == "N":
                onehotencoding.append([0,0,0,0])
        arrayoutput.append(np.array(onehotencoding))
    return np.array(arrayoutput)

# Processing the input data file
def process_input(fileinput,delimiter="\t"):
    Xset = []
    Yset = []
    global sequencelength, limiter, limiterstart, limiterend
    for line in fileinput:
        if "R900" not in line:
            line = line.split(delimiter)
            if limiter: line[0] = line[0][limiterstart:limiterend]
            if sequencelength==0:
                sequencelength=len(line[0])
            Xset.append(line[0])
            Yset.append(float(line[1].strip("\n")))
    Xset = np.array(Xset)
    Yset = np.array(Yset)
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xset, Yset,test_size=0.2,random_state=1)
    Xtrain_conv = convert_base(Xtrain)
    Xtest_conv = convert_base(Xtest)
    '''
    # Read in from file
    # Convert strings to one hot encodings
    # Return numpy arrays of the one hot encoded sequences and the corresponding y-values
    '''
    return Xtrain_conv, Xtest_conv, ytrain, ytest, Xtrain, Xtest

# Tensorflow keras machine learning models (4)
def predictor(Xtrain,Xtest,ytrain,ytest,epoch_total=5,version="CNN",verbosity=1,crossval=True):
    global sequencelength, fileout, Xtrain_raw, Xtest_raw

    i = k.Input(shape=(sequencelength,4), name="Input")
    if version.upper()=="CNN": 
        cnn = k.layers.Conv1D(128,3,padding="same", name="cnn")(i)
        cnn_a = k.layers.LeakyReLU(name="cnn_activation")(cnn)
        cnn_pool = k.layers.MaxPooling1D(pool_size=2,padding="same", name="cnn_pool")(cnn_a)
        cnn_dr = k.layers.Dropout(0.3, name="cnn_dropout")(cnn_pool)

        f = k.layers.Flatten(name="flatten")(cnn_dr)

        d1 = k.layers.Dense(128, name="dense_1")(f)
        da1 = k.layers.LeakyReLU(name="dense_activation_1")(d1)
        ddr1 = k.layers.Dropout(0.2, name="dense_dropout_1")(da1)
        d2 = k.layers.Dense(64, name="dense_2")(ddr1)
        da2 = k.layers.LeakyReLU(name="dense_activation_2")(d2)
        ddr2 = k.layers.Dropout(0.2, name="dense_dropout_2")(da2)
        o = k.layers.Dense(1, activation="linear", name="output")(ddr2)
    
    elif "CNN" not in version.upper() and "RNN" in version.upper():
        bgru_rnn = k.layers.Bidirectional(k.layers.GRU(128, kernel_initializer='he_normal', dropout=0.3, recurrent_dropout=0.2), name="bgru_rnn")(i)

        f = k.layers.Flatten(name="flatten")(bgru_rnn)

        d1 = k.layers.Dense(128, name="dense_1")(f)
        da1 = k.layers.LeakyReLU(name="dense_activation_1")(d1)
        ddr1 = k.layers.Dropout(0.2, name="dense_dropout_1")(da1)
        d2 = k.layers.Dense(64, name="dense_2")(ddr1)
        da2 = k.layers.LeakyReLU(name="dense_activation_2")(d2)
        ddr2 = k.layers.Dropout(0.2, name="dense_dropout_2")(da2)
        o = k.layers.Dense(1, activation="linear", name="output")(ddr2)

    elif "CNN" in version.upper() and "RNN" in version.upper():
        cnn = k.layers.Conv1D(128,3,padding="same", name="cnn")(i)
        cnn_a = k.layers.LeakyReLU(name="cnn_activation")(cnn)
        cnn_pool = k.layers.MaxPooling1D(pool_size=2,padding="same", name="cnn_pool")(cnn_a)
        cnn_dr = k.layers.Dropout(0.3, name="cnn_dropout")(cnn_pool)

        bgru_rnn = k.layers.Bidirectional(k.layers.GRU(64, kernel_initializer='he_normal', dropout=0.3, recurrent_dropout=0.2), name="bgru_rnn")(cnn_dr)

        f = k.layers.Flatten(name="flatten")(bgru_rnn)

        d1 = k.layers.Dense(128, name="dense_1")(f)
        da1 = k.layers.LeakyReLU(name="dense_activation_1")(d1)
        ddr1 = k.layers.Dropout(0.2, name="dense_dropout_1")(da1)
        d2 = k.layers.Dense(64, name="dense_2")(ddr1)
        da2 = k.layers.LeakyReLU(name="dense_activation_2")(d2)
        ddr2 = k.layers.Dropout(0.2, name="dense_dropout_2")(da2)
        o = k.layers.Dense(1, activation="linear", name="output")(ddr2)
    
    elif version.upper()=="LINEAR":
        f = k.layers.Flatten(name="flatten")(i)
        o = k.layers.Dense(1, activation="linear", name="output")(f)

    # Model Initiation
    m = k.Model(inputs=i, outputs=o)
    compile_options={"optimizer": "adam", "loss": "mean_squared_error"}
    m.compile(**compile_options)
    m.summary()
    m.save("untrained.h5")
    
    # Cross validation testing
    if crossval:
        outputfile = open("output_golden_gate_prediction_no_zero_CV_" + version.upper() + ".tsv","w+")
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        scores = []
        for train_index, test_index in kf.split(Xtrain):
            cvXtrain, cvXtest = Xtrain[train_index], Xtrain[test_index]
            cvytrain, cvytest = ytrain[train_index], ytrain[test_index]
            m = k.models.load_model("untrained.h5")
            m.compile(**compile_options)
            outputfile.write("NewFold\n")
            for i in range(0,epoch_total):
                m.fit(cvXtrain,cvytrain,epochs=1,batch_size=32,verbose=0)
                singleepochresult = spearmanr(m.predict(cvXtest,verbose=1),cvytest,axis=0)
                outputfile.write(str(singleepochresult[0]) + "\n")
            result = spearmanr(m.predict(cvXtest),cvytest,axis=0)
            print(result[0])
            scores.append(result[0])
        
        print("5-fold cross validation average:")
        print(str(sum(scores)/5))
        outputfile.close()

    # Test set data prediction and output (printed correlation, written file output containing prediction & true values for each sequence)
    else:
        m = k.models.load_model("untrained.h5")
        fileout.write(version + "\n")
        for i in range(0,epoch_total):
            m.fit(Xtrain,ytrain,epochs=1,batch_size=32,verbose=verbosity)
            fileout.write(str(spearmanr(m.predict(Xtest),ytest,axis=0)[0]) + "\n")

        ypred = m.predict(Xtest)
        print("Test set spearman correlation:")
        print(spearmanr(ypred,ytest,axis=0))

        outputfile = open("output_golden_gate_prediction_no_zero_test_set_" + version.upper() + ".tsv","w+")
        outputfile.write(version.upper() + "_test\ttrue\tprediction\n")
        for i in range(len(Xtest_raw)):
            outputfile.write(str(Xtest_raw[i]) + "\t" + str(ytest[i]) + "\t" + str(ypred[i]).replace('[','').replace(']','') + "\n")
        outputfile.close()

    k.backend.clear_session()
    return

# Main function which loads the data, parses it, and calls each of the 4 machine learning models
# Modify the "predictor" line's "crossval" arguments to instead run the 5-fold cross validation instead of the test set prediction
def main():
    global limiter, limiterstart, limiterend, fileout, Xtrain_raw, Xtest_raw
    limiter = True
    limiterstart = 0
    limiterend = 20
    filein = open("golden_gate_cloning_dataset.csv","r")
    Xtrainvals, Xtestvals, ytrainvals, ytestvals, Xtrain_raw, Xtest_raw = process_input(filein,",")
    print("\n\nRunning the Linear Model with 5-fold cross validation on 80 percent and testing on 20 percent of the data.\n")
    predictor(Xtrainvals, Xtestvals, ytrainvals, ytestvals,epoch_total=200,version="Linear",verbosity=0,crossval=False)
    print("\n\nRunning the CNN-Based Model with 5-fold cross validation on 80 percent and testing on 20 percent of the data.\n")
    predictor(Xtrainvals, Xtestvals, ytrainvals, ytestvals,epoch_total=38,version="CNN",verbosity=1,crossval=False)
    print("\n\nRunning the RNN-Based Model with 5-fold cross validation on 80 percent and testing on 20 percent of the data.\n")
    predictor(Xtrainvals, Xtestvals, ytrainvals, ytestvals,epoch_total=16,version="RNN",verbosity=1,crossval=False)
    print("\n\nRunning the CNN-RNN-Based Model with 5-fold cross validation on 80 percent and testing on 20 percent of the data.\n")
    predictor(Xtrainvals, Xtestvals, ytrainvals, ytestvals,epoch_total=18,version="CNNRNN",verbosity=1,crossval=False)

    filein.close()
    fileout.close()

main()
