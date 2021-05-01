import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

IMAGE_NUM = 500
WORD_NUM = 200

category = ['bo', 'chu', 'gong', 'hang', 'huai']
training_path = 'dataset/training/'
testing_path = 'dataset/testing/'
validation_path = 'dataset/validation/'
feature_path = 'dataset/surf/'
vocabulary_path = 'dataset/surf/'
svm_model_name = 'surf_random_forest.clf'


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
    sift = cv2.xfeatures2d.SURF_create(200)  # max number of SIFT points is 200
    kp, des = sift.detectAndCompute(gray, None)
    return des


def calcFeatVec(features, centers):
    featVec = np.zeros((1, WORD_NUM))
    for i in range(0, features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi, (WORD_NUM, 1)) - centers
        sqSum = (diffMat ** 2).sum(axis=1)
        dist = sqSum ** 0.5
        sortedIndices = dist.argsort()
        idx = sortedIndices[0]  # index of the nearest center
        featVec[0][idx] += 1
    return featVec


def initFeatureSet():
    training_image_name_list = os.listdir(training_path)
    featureSet = np.float32([]).reshape(0, 64)
    for filename in training_image_name_list:
        file_path = training_path + filename
        img = cv2.imread(file_path)
        des = calcSiftFeature(img)
        featureSet = np.append(featureSet, des, axis=0)
    featCnt = featureSet.shape[0]
    print(featureSet.shape)
    print(str(featCnt) + " feature ")
    filename = feature_path + "feature.npy"
    np.save(filename, featureSet)


def learnVocabulary():
    wordCnt = WORD_NUM
    filename = feature_path + "feature.npy"
    features = np.load(filename)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(features, wordCnt, None, criteria, 20, flags)
    # save vocabulary(a tuple of (labels, centers)) to file
    filename = vocabulary_path + "Vocabulary.npy"
    np.save(filename, (labels, centers))


def trainClassifier():
    trainData = np.float32([]).reshape(0, WORD_NUM)
    response = np.int32([])
    labels, centers = np.load(vocabulary_path + "Vocabulary.npy", allow_pickle=True)
    training_image_name_list = os.listdir(training_path)
    for filename in training_image_name_list:
        file_path = training_path + filename
        img = cv2.imread(file_path)
        features = calcSiftFeature(img)
        featVec = calcFeatVec(features, centers)
        trainData = np.append(trainData, featVec, axis=0)

    response = np.repeat([0, 1, 2, 3, 4], 100)
    trainData = np.float32(trainData)
    response = response.reshape(-1, 1)
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.train_auto(trainData, response, None, None, None) # select best params
    svm.train(trainData, cv2.ml.ROW_SAMPLE, response)
    svm.save(svm_model_name)


def train_rtree_classifier():
    trainData = np.float32([]).reshape(0, WORD_NUM)
    labels, centers = np.load(vocabulary_path + "Vocabulary.npy", allow_pickle=True)
    training_image_name_list = os.listdir(training_path)
    for filename in training_image_name_list:
        file_path = training_path + filename
        img = cv2.imread(file_path)
        features = calcSiftFeature(img)
        featVec = calcFeatVec(features, centers)
        trainData = np.append(trainData, featVec, axis=0)

    response = np.repeat([0, 1, 2, 3, 4], 100)
    trainData = np.float32(trainData)
    response = response.reshape(-1, 1)
    random_forest = RandomForestClassifier(n_estimators=150, max_depth= 20, min_samples_split=5)
    random_forest.fit(trainData, response)
    return random_forest


def random_forest_classify(random_forest):

    total = 0
    correct = 0
    dictIdx = 0
    labels, centers = np.load(vocabulary_path + "Vocabulary.npy", allow_pickle=True)
    for name, index_range in TestSetInfo.items():
        count = index_range[1] + 1 - index_range[0]
        crt = 0
        for i in range(index_range[0], index_range[1] + 1):
            filename = validation_path + name + '_' + str(i) + '.png'
            img = cv2.imread(filename)
            features = calcSiftFeature(img)
            featVec = calcFeatVec(features, centers)
            case = np.float32(featVec)
            # print(case)
            pre = random_forest.predict(case)
            # print(pre)
            # pre = int(pre[1])

            if dictIdx == pre[0]:
                crt += 1

        print("Accuracy: " + str(crt) + " / " + str(count) + "\n")
        total += count
        correct += crt
        dictIdx += 1

    print("Total accuracy: " + str(correct) + " / " + str(total))

def classify():
    # svm = cv2.SVM()
    # svm = cv2.ml.SVM_create()
    svm = cv2.ml.SVM_load(svm_model_name)

    total = 0
    correct = 0
    dictIdx = 0
    labels, centers = np.load(vocabulary_path + "Vocabulary.npy", allow_pickle=True)
    for name, index_range in TestSetInfo.items():
        count = index_range[1] + 1 - index_range[0]
        crt = 0
        for i in range(index_range[0], index_range[1] + 1):
            filename = validation_path + name + '_' + str(i) + '.png'
            img = cv2.imread(filename)
            features = calcSiftFeature(img)
            featVec = calcFeatVec(features, centers)
            case = np.float32(featVec)
            # print(case)
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
    initFeatureSet()
    learnVocabulary()
    # trainClassifier()

    random_forest_classify(train_rtree_classifier())
