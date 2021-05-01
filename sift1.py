import cv2
import numpy as np
import os

IMAGE_NUM = 500

category = ['bo', 'chu', 'gong', 'hang', 'huai']
training_path = 'dataset/training/'
testing_path = 'dataset/testing/'
validation_path = 'dataset/validation/'
feature_path = 'dataset/sift_feature/'
vocabulary_path = 'dataset/sift_vocabulary/'
svm_model_name = 'sift_svm.clf'

TrainSetInfo = {
    'bo': (1, 100),
    'chu': (101, 200),
    'gong': (201, 300),
    'hang': (301, 400),
    'huai': (401, 500)
}

TestSetInfo = {
    'bo': (161, 200),
    'chu': (1, 40),
    'gong': (41, 80),
    'hang': (81, 120),
    'huai': (121, 160)
}


def calcSiftFeature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(200)  # max number of SIFT points is 200
    kp, des = sift.detectAndCompute(gray, None)
    return des


def calcFeatVec(features, centers):
    featVec = np.zeros((1, 50))
    for i in range(0, features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi, (50, 1)) - centers
        sqSum = (diffMat ** 2).sum(axis=1)
        dist = sqSum ** 0.5
        sortedIndices = dist.argsort()
        idx = sortedIndices[0]  # index of the nearest center
        featVec[0][idx] += 1
    return featVec


def initFeatureSet():
    for name, index_range in TrainSetInfo.items():

        featureSet = np.float32([]).reshape(0, 128)
        for i in range(index_range[0], index_range[1] + 1):
            filename = training_path + name + '_' + str(i) + '.png'
            img = cv2.imread(filename)
            des = calcSiftFeature(img)
            featureSet = np.append(featureSet, des, axis=0)
        featCnt = featureSet.shape[0]
        print(featureSet.shape)
        print(str(featCnt) + " feature ")
        filename = feature_path + name + ".npy"
        np.save(filename, featureSet)


def learnVocabulary():
    wordCnt = 50
    for name, index_range in TrainSetInfo.items():
        filename = feature_path + name + ".npy"
        features = np.load(filename)

        print("Learn vocabulary of " + name + "...")
        # use k-means to cluster a bag of features
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(features, wordCnt, None, criteria, 20, flags)

        # save vocabulary(a tuple of (labels, centers)) to file
        filename = vocabulary_path + name + ".npy"
        np.save(filename, (labels, centers))
        print("Done\n")


def trainClassifier():
    trainData = np.float32([]).reshape(0, 50)
    response = np.int32([])

    dictIdx = 0
    for name, index_range in TrainSetInfo.items():
        # dir = "D:/GJAI_data/gongjingai_pre/train_1-2-3/" + name + "/"
        print(vocabulary_path + name + ".npy")
        labels, centers = np.load(vocabulary_path + name + ".npy", allow_pickle=True)

        print("Init training data of " + name + "...")
        for i in range(index_range[0], index_range[1] + 1):
            filename = training_path + name + '_' + str(i) + '.png'
            img = cv2.imread(filename)
            features = calcSiftFeature(img)
            featVec = calcFeatVec(features, centers)
            trainData = np.append(trainData, featVec, axis=0)

        res = np.repeat(dictIdx, index_range[1] + 1 - index_range[0])
        response = np.append(response, res)
        dictIdx += 1
        print("Done\n")
    print(response)
    print("Now train svm classifier...")
    trainData = np.float32(trainData)
    response = response.reshape(-1, 1)
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.train_auto(trainData, response, None, None, None) # select best params
    svm.train(trainData, cv2.ml.ROW_SAMPLE, response)
    svm.save(svm_model_name)
    print("Done\n")


def trainClassifier():
    trainData = np.float32([]).reshape(0, 50)
    response = np.int32([])

    dictIdx = 0
    for name, index_range in TrainSetInfo.items():
        # dir = "D:/GJAI_data/gongjingai_pre/train_1-2-3/" + name + "/"
        print(vocabulary_path + name + ".npy")
        labels, centers = np.load(vocabulary_path + name + ".npy", allow_pickle=True)

        print("Init training data of " + name + "...")
        for i in range(index_range[0], index_range[1] + 1):
            filename = training_path + name + '_' + str(i) + '.png'
            img = cv2.imread(filename)
            features = calcSiftFeature(img)
            featVec = calcFeatVec(features, centers)
            trainData = np.append(trainData, featVec, axis=0)

        res = np.repeat(dictIdx, index_range[1] + 1 - index_range[0])
        response = np.append(response, res)
        dictIdx += 1
        print("Done\n")
    print(response)
    print("Now train svm classifier...")
    trainData = np.float32(trainData)
    response = response.reshape(-1, 1)
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.train_auto(trainData, response, None, None, None) # select best params
    svm.train(trainData, cv2.ml.ROW_SAMPLE, response)
    svm.save(svm_model_name)
    print("Done\n")


def classify():
    # svm = cv2.SVM()
    # svm = cv2.ml.SVM_create()
    svm = cv2.ml.SVM_load(svm_model_name)

    total = 0
    correct = 0
    dictIdx = 0
    for name, index_range in TestSetInfo.items():
        count = index_range[1] + 1 - index_range[0]
        crt = 0
        # dir = "D:/GJAI_data/gongjingai_pre/validation_1-2-3/" + name + "/"

        labels, centers = np.load(vocabulary_path + name + ".npy",allow_pickle=True)

        for i in range(index_range[0], index_range[1] + 1):
            filename = validation_path + name + '_' + str(i) + '.png'
            img = cv2.imread(filename)
            features = calcSiftFeature(img)
            featVec = calcFeatVec(features, centers)
            case = np.float32(featVec)
            print(case)
            dict_svm = svm.predict(case)
            dict_svm = int(dict_svm[1])
            if dictIdx == dict_svm:
                crt += 1

        print("Accuracy: " + str(crt) + " / " + str(count) + "\n")
        total += count
        correct += crt
        dictIdx += 1

    print("Total accuracy: " + str(correct) + " / " + str(total))


def classify2():
    # svm = cv2.SVM()
    # svm = cv2.ml.SVM_create()
    svm = cv2.ml.SVM_load(svm_model_name)

    total = 0
    correct = 0
    dictIdx = 0
    validation_image_list = os.listdir(validation_path)
    print(validation_image_list)
    for name, index_range in TestSetInfo.items():
        count = index_range[1] + 1 - index_range[0]
        crt = 0
        # dir = "D:/GJAI_data/gongjingai_pre/validation_1-2-3/" + name + "/"

        labels, centers = np.load(vocabulary_path + name + ".npy", allow_pickle=True)

        for i in range(index_range[0], index_range[1] + 1):
            filename = validation_path + name + '_' + str(i) + '.png'
            img = cv2.imread(filename)
            features = calcSiftFeature(img)
            featVec = calcFeatVec(features, centers)
            case = np.float32(featVec)
            dict_svm = svm.predict(case)
            dict_svm = int(dict_svm[1])
            if dictIdx == dict_svm:
                crt += 1

        print("Accuracy: " + str(crt) + " / " + str(count) + "\n")
        total += count
        correct += crt
        dictIdx += 1

    print("Total accuracy: " + str(correct) + " / " + str(total))

if __name__ == "__main__":
    # initFeatureSet()
    # learnVocabulary()
    trainClassifier()
    # classify()
