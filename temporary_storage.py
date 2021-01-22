from abc import ABC
from abc import abstractmethod

class TemporaryStorage(ABC):
    """
    A class that represents a temporary storage for training and inference data.

    Attributes
    ----------
    name : region to which data corresponds
    positions: an array of positions
    X : an array of examples/features
    """

    def __init__(self, name):
        """
        Parameters
        ----------
        name : region to which data corresponds
        """

        self.name = name
        self.positions = []
        self.X = []

    @abstractmethod
    def store(self, args):
        """
        Stores new data to temporary storage.
        """

        pass

    def clear(self):
        """
        Cleares storage of all data.
        """

        del self.positions[:]
        del self.X[:]

    def get_positions(self):
        """
        Gets stored positions.

        Returns
        -------
        positions : stored positions
        """

        return self.positions

    def get_X(self):
        """
        Gets stored examples.

        Returns
        -------
        X : stored examples
        """

        return self.X
    
    @abstractmethod
    def get_Y(self):
        """
        Gets stored labels.

        Returns
        -------
        Y : labels
        """

        pass

class TemporaryTrainStorage(TemporaryStorage):
    """
    A class that respresents temporary storage for training data.

    Attributes
    ----------
    Y : an array of labels
    """

    def __init__(self, name):
        super().__init__(name)
        self.Y = []

    def store(self, args):
        positions, X, Y = args

        assert Y is not None
        assert len(positions) == len(X) == len(Y)

        for i, position in enumerate(positions):
            self.positions.append(position)
            self.X.append(X[i])
            self.Y.append(Y[i])

    def clear(self):
        super().clear()
        del self.Y[:]

    def get_Y(self):
        return self.Y

class TemporaryInferenceStorage(TemporaryStorage):
    """
    A class that represents temporary storage for inference data.
    """

    def __init__(self, name):
        super().__init__(name)

    def store(self, args):
        positions, X = args

        assert len(positions) == len(X)

        for i, position in enumerate(positions):
            self.positions.append(position)
            self.X.append(X[i])

    def get_Y(self):
        return None