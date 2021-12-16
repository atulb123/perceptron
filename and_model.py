from utils.model_utility import ModelUtility
from utils.model import Model
import pandas as pd
import logging
import os

logging_str = "[%(asctime)s:%(levelname)s:%(module)s:%(message)s]"
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format=logging_str,
                    filename=os.path.join("logs", "logs.log"))
if __name__ == "__main__":
    AND = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 0, 0, 1]}
    df = pd.DataFrame(AND)
    logging.info(f"This is the dataframe :{df}")
    model_utility = ModelUtility()
    x, y = model_utility.prepare_data(df)
    model = Model(0.3, 10)
    model.fit(x, y)
    logging.info(f"predicted values are: {model.predict(x)}")
    model_utility.save_model(model, "and_model.pickle")
    model_utility.save_plot(df, model, "and_plot.png")
