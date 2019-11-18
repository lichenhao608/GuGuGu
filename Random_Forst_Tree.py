import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data import load_data


if __name__ == "__main__":
    data, y = load_data('data/train_labels.csv', 'data/train/', num=50000)

    model = RandomForestClassifier()
    effect_area = data[:, 32:64, 32:64]
    effect_area = effect_area.reshape((50000, -1))
    train, test, y_train, y_test = train_test_split(
        effect_area, y, train_size=0.8)

    model.fit(train, y_train)
    model.score(test, y_test)
