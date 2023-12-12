from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib


class ModelTraining:
    """
    A class for training, evaluating, and saving machine learning models.
    """

    def __init__(self, model, data, target_column):
        """
        Initialize the ModelTraining instance.

        Parameters:
            model: The machine learning model to be trained.
            data (pandas.DataFrame): The dataset used for training and evaluation.
            target_column (str): The name of the target column in the dataset.
        """
        self.model = model
        self.data = data
        self.target_column = target_column

    def split_data(self, test_size=0.3, random_state=None):
        """
        Split the dataset into training and testing sets.

        Parameters:
            test_size (float, optional): The proportion of the dataset to include in the test split.
            random_state (int, optional): Seed for the random number generator.

        Returns:
            None
        """
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

    def train_model(self):
        """
        Train the machine learning model using the training set.

        Returns:
            None
        """
        self.model.fit(self.X_train, self.y_train)

    def predict_new_data(self, new_data):
        """
        Make predictions on new data using the trained model.

        Parameters:
            new_data (pandas.DataFrame): New data for prediction.

        Returns:
            numpy.ndarray: Predicted values.
        """
        return self.model.predict(new_data)

    def evaluate_model(self, target_names=None):
        """
        Evaluate the model's performance on the test set.

        Parameters:
            target_names (list, optional): List of target class names for classification report.

        Returns:
            str: Classification report.
        """
        y_pred = self.model.predict(self.X_test)
        return classification_report(self.y_test, y_pred, target_names=target_names)

    def hyperparameter_tuning(self, param_grid, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV.

        Parameters:
            param_grid (dict): Dictionary of hyperparameter values to search.
            cv (int, optional): Number of cross-validation folds.

        Returns:
            None
        """
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_

    @staticmethod
    def build_filename(model_name):
        """
        Build a filename for saving the trained model.

        Parameters:
            model_name (str): Name of the model.

        Returns:
            str: Complete filename.
        """
        return f"modelos/{model_name}.joblib"

    def save_model(self, model_name='initial'):
        """
        Save the trained model to a file.

        Parameters:
            model_name (str, optional): Name of the model for the filename.

        Returns:
            None
        """
        joblib.dump(self.model, filename=self.build_filename(model_name))

    def load_model(self, model_name='initial'):
        """
        Load a pre-trained model from a file.

        Parameters:
            model_name (str, optional): Name of the model for the filename.

        Returns:
            None
        """
        self.model = joblib.load(filename=self.build_filename(model_name))

    def feature_importance(self):
        """
        Retrieve feature importances from the trained model.

        Returns:
            numpy.ndarray: Feature importances (if applicable).
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            print(
                f"Este modelo ({str(type(self.model))}) não suporta análise de importância de recursos.")
