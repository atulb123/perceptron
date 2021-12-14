from utils.model_utility import ModelUtility
from utils.model import Model
import pandas as pd

if __name__ == "__main__":
    AND = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 0, 0, 1]}
    df = pd.DataFrame(AND)
    model_utility = ModelUtility()
    x, y = model_utility.prepare_data(df)
    model = Model(0.3, 10)
    model.fit(x, y)
    print(f"predicted values are: {model.predict(x)}")
    model_utility.save_model(model, "and_model.pickle")
    model_utility.save_plot(df, model, "and_plot.png")
