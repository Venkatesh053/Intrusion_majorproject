import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_distribution(df):
    """Class distribution of NSL-KDD"""
    fig, ax = plt.subplots()
    sns.countplot(x="labels", data=df, ax=ax)
    ax.set_title("Normal vs Attack Distribution")
    return fig

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues")
    return fig
