import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import logging

class ModelUtility:
    """"""
    def prepare_data(self, df):
        X = df.drop("y", axis=1)
        y = df["y"]
        return X, y

    def save_model(self, model, filename):
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)  # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
        filePath = os.path.join(model_dir, filename)  # model/filename
        joblib.dump(model, filePath)

    def save_plot(self, df, model, file_name):
        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(10, 8)
        colors = ("red", "blue", "lightgreen", "gray", "cyan")
        x, y = self.prepare_data(df)
        cmap = ListedColormap(colors[: len(np.unique(y))])
        x = x.values  # as a array
        x1 = x[:, 0]
        x2 = x[:, 1]
        x1_min, x1_max = x1.min() - 1, x1.max() + 1
        x2_min, x2_max = x2.min() - 1, x2.max() + 1

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                               np.arange(x2_min, x2_max, 0.02))
        Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        os.makedirs("plots", exist_ok=True)
        plt.plot()
        plt.savefig(os.path.join("plots", file_name))
