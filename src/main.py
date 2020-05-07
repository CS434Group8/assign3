from tree import DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier
from utils import load_data, f1, accuracy_score, load_dictionary, dictionary_info, get_one_dictionary_info
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
sns.set()


def load_args():

    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--county_dict', default=1, type=int)
    parser.add_argument('--decision_tree', default=1, type=int)
    parser.add_argument('--random_forest', default=1, type=int)
    parser.add_argument('--ada_boost', default=1, type=int)
    parser.add_argument('--root_dir', default='../data/', type=str)
    args = parser.parse_args()

    return args


def county_info(args):
    county_dict = load_dictionary(args.root_dir)
    dictionary_info(county_dict)


def decision_tree_testing(x_train, y_train, x_test, y_test):
    print('Decision Tree\n\n')
    clf = DecisionTreeClassifier(max_depth=20)
    clf.fit(x_train, y_train)
    preds_train = clf.predict(x_train)
    preds_test = clf.predict(x_test)
    train_accuracy = accuracy_score(preds_train, y_train)
    test_accuracy = accuracy_score(preds_test, y_test)
    print('Train {}'.format(train_accuracy))
    print('Test {}'.format(test_accuracy))
    preds = clf.predict(x_test)
    print('F1 Test {}'.format(f1(y_test, preds)))


def decision_tree_tune(x_train, y_train, x_test, y_test):
    print('Decision Tree tune\n\n')
    plotX = [i for i in range(1, 26)]
    plotTrain = []
    plotTest = []
    plotF1 = []

    for depth in range(1, 26):
        print('Math Depth: ', depth)
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(x_train, y_train)
        preds_train = clf.predict(x_train)
        preds_test = clf.predict(x_test)
        train_accuracy = round(accuracy_score(preds_train, y_train), 3)
        test_accuracy = round(accuracy_score(preds_test, y_test), 3)
        print('Train {}'.format(train_accuracy))
        print('Test {}'.format(test_accuracy))
        preds = clf.predict(x_test)
        F1 = round(f1(y_test, preds), 3)
        print('F1 Test {}'.format(F1))
        print('\n')
        plotTrain.append(train_accuracy)
        plotTest.append(test_accuracy)
        plotF1.append(F1)

    df = pd.DataFrame({"Max_Depth": plotX, "Train_Accuracy": plotTrain,
                       "Test_Accuracy": plotTest, "F1_Accuracy": plotF1})
    print(df)
    maxAccuracy = max(plotF1)
    bestDepth = plotX[plotF1.index(maxAccuracy)]
    print("The best Depth is ", bestDepth, "with F1 accuracy ", maxAccuracy)

    print("Drawing plot")
    plt.plot('Max_Depth', 'Train_Accuracy', data=df, color='red')
    plt.plot('Max_Depth', 'Test_Accuracy', data=df, color='blue')
    plt.plot('Max_Depth', 'F1_Accuracy', data=df, color='black')
    plt.legend()
    plt.savefig('decision_tree_output.png')
    plt.close()
    return bestDepth


def find_most_important_feature(x_train, y_train, depth):
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(x_train, y_train)
    return clf.root.feature


def random_forest_tune_MaxFeatures(x_train, y_train, x_test, y_test):
    print('Random Forest tune\n\n')
    plotX = [1, 2, 5, 8, 10, 20, 25, 35, 50]
    plotTrain = []
    plotTest = []
    plotF1 = []

    for max_features in plotX:
        print("MAX_Features: ", max_features)
        rclf = RandomForestClassifier(
            max_depth=7, max_features=max_features, n_trees=50)
        rclf.fit(x_train, y_train)
        preds_train = rclf.predict(x_train)
        preds_test = rclf.predict(x_test)
        train_accuracy = round(accuracy_score(preds_train, y_train), 3)
        test_accuracy = round(accuracy_score(preds_test, y_test), 3)
        print('Train {}'.format(train_accuracy))
        print('Test {}'.format(test_accuracy))
        preds = rclf.predict(x_test)
        F1 = round(f1(y_test, preds), 3)
        print('F1 Test {}'.format(F1))
        print('\n')
        plotTrain.append(train_accuracy)
        plotTest.append(test_accuracy)
        plotF1.append(F1)

    df = pd.DataFrame({"MAX_Features": plotX, "Train_Accuracy": plotTrain,
                       "Test_Accuracy": plotTest, "F1_Accuracy": plotF1})
    print(df)
    maxAccuracy = max(plotF1)
    best_MAX_Features = plotX[plotF1.index(maxAccuracy)]
    print("The best MAX_Features is ", best_MAX_Features,
          "with F1 accuracy ", maxAccuracy)

    print("Drawing plot")
    plt.plot('MAX_Features', 'Train_Accuracy', data=df, color='red')
    plt.plot('MAX_Features', 'Test_Accuracy', data=df, color='blue')
    plt.plot('MAX_Features', 'F1_Accuracy', data=df, color='black')
    plt.legend()
    plt.savefig('random_forest_output_max_features.png')
    plt.close()
    return best_MAX_Features


def random_forest_tune_NTree(x_train, y_train, x_test, y_test):
    print('Random Forest tune\n\n')
    plotX = [i for i in range(10, 210, 10)]
    plotTrain = []
    plotTest = []
    plotF1 = []

    for n_trees in plotX:
        print("N_Trees: ", n_trees)
        rclf = RandomForestClassifier(
            max_depth=7, max_features=11, n_trees=n_trees)
        rclf.fit(x_train, y_train)
        preds_train = rclf.predict(x_train)
        preds_test = rclf.predict(x_test)
        train_accuracy = round(accuracy_score(preds_train, y_train), 3)
        test_accuracy = round(accuracy_score(preds_test, y_test), 3)
        print('Train {}'.format(train_accuracy))
        print('Test {}'.format(test_accuracy))
        preds = rclf.predict(x_test)
        F1 = round(f1(y_test, preds), 3)
        print('F1 Test {}'.format(F1))
        print('\n')
        plotTrain.append(train_accuracy)
        plotTest.append(test_accuracy)
        plotF1.append(F1)

    df = pd.DataFrame({"N_Trees": plotX, "Train_Accuracy": plotTrain,
                       "Test_Accuracy": plotTest, "F1_Accuracy": plotF1})
    print(df)
    maxAccuracy = max(plotF1)
    best_N_Trees = plotX[plotF1.index(maxAccuracy)]
    print("The best N_Trees is ", best_N_Trees,
          "with F1 accuracy ", maxAccuracy)

    print("Drawing plot")
    plt.plot('N_Trees', 'Train_Accuracy', data=df, color='red')
    plt.plot('N_Trees', 'Test_Accuracy', data=df, color='blue')
    plt.plot('N_Trees', 'F1_Accuracy', data=df, color='black')
    plt.legend()
    plt.savefig('random_forest_output_Ntrees.png')
    plt.close()
    return best_N_Trees


def random_forest_testing(x_train, y_train, x_test, y_test):
    print('Random Forest\n\n')
    rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=50)
    rclf.fit(x_train, y_train)
    preds_train = rclf.predict(x_train)
    preds_test = rclf.predict(x_test)
    train_accuracy = accuracy_score(preds_train, y_train)
    test_accuracy = accuracy_score(preds_test, y_test)
    print('Train {}'.format(train_accuracy))
    print('Test {}'.format(test_accuracy))
    preds = rclf.predict(x_test)
    print('F1 Test {}'.format(f1(y_test, preds)))


def random_forest_seed_testing(x_train, y_train, x_test, y_test):
    print('Random Forest seed testing\n\n')

    rclf = RandomForestClassifier(max_depth=15, max_features=8, n_trees=210)

    F1_result = []
    Train_result = []
    Test_result = []
    seeds = [x for x in range(1, 100, 10)]
    for seed in seeds:
        rclf.seed = seed
        rclf.fit(x_train, y_train)
        preds_train = rclf.predict(x_train)
        preds_test = rclf.predict(x_test)
        train_accuracy = round(accuracy_score(preds_train, y_train), 3)
        test_accuracy = round(accuracy_score(preds_test, y_test), 3)
        print('Train {}'.format(train_accuracy))
        print('Test {}'.format(test_accuracy))
        preds = rclf.predict(x_test)
        F1 = round(f1(y_test, preds), 3)
        print('F1 Test {}'.format(F1))
        print('\n')
        F1_result.append(F1)
        Train_result.append(train_accuracy)
        Test_result.append(test_accuracy)

    seeds.append("Average")
    F1_result.append(sum(F1_result)/len(F1_result))
    Train_result.append(sum(Train_result)/len(Train_result))
    Test_result.append(sum(Test_result)/len(Test_result))
    df = pd.DataFrame({"Seed": seeds, "F1": F1_result,
                       "Train": Train_result, "Test": Test_result})
    print(df)


def rf_tune_all(x_train, y_train, x_test, y_test, depth, features, trees):
    print('Random Forest tune 3 parameters\n\n')
    print("[depth,features,trees]")
    print([depth, features, trees])
    rclf = RandomForestClassifier(
        max_depth=depth, max_features=features, n_trees=trees)
    rclf.fit(x_train, y_train)
    preds_train = rclf.predict(x_train)
    preds_test = rclf.predict(x_test)
    train_accuracy = round(accuracy_score(preds_train, y_train), 3)
    test_accuracy = round(accuracy_score(preds_test, y_test), 3)
    print('Train {}'.format(train_accuracy))
    print('Test {}'.format(test_accuracy))
    preds = rclf.predict(x_test)
    F1 = round(f1(y_test, preds), 3)
    print('F1 Test {}'.format(F1))
    print('\n')
    return F1


def adaBoost_testing(x_train, y_train, x_test, y_test, l):

    clf = AdaBoostClassifier(l)
    clf.fit(x_train, y_train)

    preds_train = clf.predict(x_train)
    preds_test = clf.predict(x_test)
    train_accuracy = round(accuracy_score(preds_train, y_train), 3)
    test_accuracy = round(accuracy_score(preds_test, y_test), 3)
    print("L: ", l)
    print('Train {}'.format(train_accuracy))
    print('Test {}'.format(test_accuracy))
    preds = clf.predict(x_test)
    F1 = round(f1(y_test, preds), 3)
    print('F1 Test {}'.format(F1))
    print()
    return train_accuracy, test_accuracy, F1


###################################################
# Modify for running your experiments accordingly #
###################################################
if __name__ == '__main__':
    args = load_args()
    x_train, y_train, x_test, y_test = load_data(args.root_dir)
    if args.county_dict == 1:
        county_info(args)
    if args.decision_tree == 1:
        print("Decision Tree")
        decision_tree_testing(x_train, y_train, x_test, y_test)
        print('\n')
        bestDepth = decision_tree_tune(x_train, y_train, x_test, y_test)
        feature_index = find_most_important_feature(
            x_train, y_train, bestDepth)
        county_dict = load_dictionary(args.root_dir)
        get_one_dictionary_info(feature_index, county_dict)
    if args.random_forest == 1:
        print("Random Forest. It takes time for tune Ntree,MaxFeatures,Seed testing")
        random_forest_testing(x_train, y_train, x_test, y_test)
        random_forest_tune_NTree(x_train, y_train, x_test, y_test)
        random_forest_tune_MaxFeatures(x_train, y_train, x_test, y_test)
        # random_forest_seed_testing(x_train, y_train, x_test, y_test)

        # pool = mp.Pool(mp.cpu_count())

        # Ntree_Task=[x for x in range(150,300,10)]
        # Max_features_Task=[5,6,7,8,9,10,11,12]
        # max_depth_Task=[10,12,14,16,18,20]

        # all_tasks=[(x_train, y_train, x_test, y_test,x,y,z) for x in Max_features_Task for y in Ntree_Task for z in max_depth_Task]

        # result=pool.starmap(rf_tune_all,all_tasks)
        # print(all_tasks)
        # print(result)

        # bestAccuracy=max(result)
        # bestcombo=all_tasks[result.index(bestAccuracy)]

        # print("The best combo of (Features,Ntree,Depth) is:")
        # print(bestcombo[-3],bestcombo[-2],bestcombo[-1])
        # print(bestAccuracy)
        # pool.close()
    if args.ada_boost == 1:
        print("AdaBoost. I am using parallel programmming, may takes 1 min")
        y_train[y_train == 0] = -1
        y_test[y_test == 0] = -1

        adaBoost_testing(x_train, y_train, x_test, y_test, 10)

        pool = mp.Pool(mp.cpu_count())
        L_Task = [x for x in range(10, 210, 10)]

        all_tasks = [(x_train, y_train, x_test, y_test, l) for l in L_Task]

        result = pool.starmap(adaBoost_testing, all_tasks)
        # print(L_Task)
        # print(result)

        pool.close()

        Train_result = []
        Test_result = []
        F1_result = []
        for r in result:
            Train_result.append(r[0])
            Test_result.append(r[1])
            F1_result.append(r[2])

        df = pd.DataFrame({"L": L_Task, "Train_Accuracy": Train_result,
                           "Test_Accuracy": Test_result, "F1_Accuracy": F1_result})

        BestAccuracy = max(F1_result)
        BestL = L_Task[F1_result.index(BestAccuracy)]
        print(df)
        print("The best L is ", BestL, " with the best accuracy: ", BestAccuracy)
        print("Drawing plot")
        plt.plot('L', 'Train_Accuracy', data=df, color='red')
        plt.plot('L', 'Test_Accuracy', data=df, color='blue')
        plt.plot('L', 'F1_Accuracy', data=df, color='black')
        plt.legend()
        plt.savefig('adaBoost_output_L.png')
        plt.close()

    print('Done')
