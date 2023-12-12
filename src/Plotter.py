import seaborn as sns
import matplotlib.pyplot as plt


class Visualization:
    """
    A class for generating various types of visualizations using Seaborn and Matplotlib.
    """

    def lineplot(self, data, x, y, hue=None, title=None, markers=None, figsize=(10, 6)):
        """
        Generate a line plot.

        Parameters:
            data (pandas.DataFrame): The dataset.
            x (str): The column name for the x-axis.
            y (str): The column name for the y-axis.
            hue (str, optional): The column name to differentiate lines by color (if applicable).
            title (str, optional): The title of the plot.
            markers (str, optional): Marker style for points in the plot.
            figsize (tuple, optional): The size of the figure (width, height).

        Returns:
            None
        """
        plt.figure(figsize=figsize)
        sns.lineplot(x=x, y=y, hue=hue, data=data, markers=markers)
        plt.title(title)
        plt.show()

    def countplot(self, data, x, hue=None, title=None, figsize=(10, 6)):
        """
        Generate a count plot.

        Parameters:
            data (pandas.DataFrame): The dataset.
            x (str): The column name for the x-axis.
            hue (str, optional): The column name to differentiate bars by color (if applicable).
            title (str, optional): The title of the plot.
            figsize (tuple, optional): The size of the figure (width, height).

        Returns:
            None
        """
        plt.figure(figsize=figsize)
        sns.countplot(x=x, hue=hue, data=data)
        plt.title(title)
        plt.show()

    def histplot(self, data, column, title=None, kde=True, figsize=(10, 6)):
        """
        Generate a histogram.

        Parameters:
            data (pandas.DataFrame): The dataset.
            column (str): The column name for the histogram.
            title (str, optional): The title of the plot.
            kde (bool, optional): Whether to include a kernel density estimate.
            figsize (tuple, optional): The size of the figure (width, height).

        Returns:
            None
        """
        plt.figure(figsize=figsize)
        bins = self.freedman_diaconis_bins(column)
        sns.histplot(data[column], kde=kde, bins=bins)
        plt.title(title)
        plt.show()

    def barplot(self, data, x, y, hue=None, title=None, figsize=(10, 6)):
        """
        Generate a bar plot.

        Parameters:
            data (pandas.DataFrame): The dataset.
            x (str): The column name for the x-axis.
            y (str): The column name for the y-axis.
            hue (str, optional): The column name to differentiate bars by color (if applicable).
            title (str, optional): The title of the plot.
            figsize (tuple, optional): The size of the figure (width, height).

        Returns:
            None
        """
        plt.figure(figsize=figsize)
        sns.barplot(x=x, y=y, hue=hue, data=data)
        plt.title(title)
        plt.show()

    def boxplot(self, data, x, y, title=None, figsize=(10, 6)):
        """
        Generate a box plot.

        Parameters:
            data (pandas.DataFrame): The dataset.
            x (str): The column name for the x-axis.
            y (str): The column name for the y-axis.
            title (str, optional): The title of the plot.
            figsize (tuple, optional): The size of the figure (width, height).

        Returns:
            None
        """
        plt.figure(figsize=figsize)
        sns.boxplot(x=x, y=y, data=data)
        plt.title(title)
        plt.show()

    def scatterplot(self, data, x, y, hue=None, title=None, figsize=(10, 6)):
        """
        Generate a scatter plot.

        Parameters:
            data (pandas.DataFrame): The dataset.
            x (str): The column name for the x-axis.
            y (str): The column name for the y-axis.
            hue (str, optional): The column name to differentiate points by color (if applicable).
            title (str, optional): The title of the plot.
            figsize (tuple, optional): The size of the figure (width, height).

        Returns:
            None
        """
        plt.figure(figsize=figsize)
        sns.scatterplot(x=x, y=y, hue=hue, data=data)
        plt.title(title)
        plt.show()

    def heatmap(self, data, title=None, figsize=(10, 6)):
        """
        Generate a heatmap.

        Parameters:
            data (pandas.DataFrame): The dataset.
            title (str, optional): The title of the plot.
            figsize (tuple, optional): The size of the figure (width, height).

        Returns:
            None
        """
        plt.figure(figsize=figsize)
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(title)
        plt.show()

    def freedman_diaconis_bins(self, data, column):
        """
        Calculate the number of bins for a histogram using the Freedman-Diaconis rule.

        Parameters:
            data (pandas.DataFrame): The dataset.
            column (str): The column name for which to calculate bins.

        Returns:
            int: The number of bins.
        """
        iqr = data[column].quantile(
            0.75) - data[column].quantile(0.25)
        h = 2 * iqr / data.shape[0]**(1/3)
        num_bins = int((data[column].max() - data[column].min()) / h)
        return num_bins
