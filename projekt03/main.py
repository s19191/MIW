import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from plotkab import plot_decision_regions
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier

def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # # standardyzacja cech
    # sc = StandardScaler()
    # sc.fit(X_train)
    # X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)
    # X_combined_std = np.vstack((X_train_std, X_test_std))
    # y_combined = np.hstack((y_train, y_test))

    giniTree01 = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    giniTree01.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=giniTree01, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.title('Gini, depth 4')
    plt.show()

    giniTree02 = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=1)
    giniTree02.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=giniTree02, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.title('Gini, depth 6')
    plt.show()

    giniTree03 = DecisionTreeClassifier(criterion='gini', max_depth=8, random_state=1)
    giniTree03.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=giniTree03, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.title('Gini, depth 8')
    plt.show()

    entropyTree01 = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)
    entropyTree01.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=entropyTree01, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.title('Entropy, depth 4')
    plt.show()

    entropyTree02 = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=1)
    entropyTree02.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=entropyTree02, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.title('Entropy, depth 6')
    plt.show()

    # export_graphviz(giniTree01, out_file='drzewo.dot', feature_names=['Długość płatka', 'Szerokość płatka'])

    forest01 = RandomForestClassifier(criterion='gini', n_estimators=1, random_state=1, n_jobs=2)
    forest01.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=forest01, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.title('Random Forest, n 1')
    plt.show()

    forest02 = RandomForestClassifier(criterion='gini', n_estimators=5, random_state=1, n_jobs=-1)
    forest02.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=forest02, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.title('Random Forest, n 5')
    plt.show()

    forest03 = RandomForestClassifier(criterion='gini', n_estimators=10, random_state=1, n_jobs=-1)
    forest03.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=forest03, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.title('Random Forest, n 10')
    plt.show()

    forest04 = RandomForestClassifier(criterion='gini', n_estimators=15, random_state=1, n_jobs=-1)
    forest04.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=forest04, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.title('Random Forest, n 15')
    plt.show()

if __name__ == '__main__':
    main()