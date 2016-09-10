from sklearn.svm import SVC
import pandas
import matplotlib.pyplot as plt


def main():
    csv = pandas.read_csv("svm-data.csv", header=None)
    y = csv[[0]]
    x = csv[[1, 2]]
    svc = SVC(C=100000, random_state=241)
    svc.fit(x, y)
    print(svc.support_)
    plt.scatter(x[1], x[2], c=y)
    plt.show()


if __name__ == '__main__':
    main()
